"""
gpt_solver.py | Author: Gabe Grand.

Queries GPT to solve tasks.

"""
import json
import os
from collections import defaultdict

from openai.error import InvalidRequestError
from openai.openai_object import OpenAIObject

import src.models.model_loaders as model_loaders
from src.experiment_iterator import RANDOM_GENERATOR, SKIPPED_MODEL_FN
from src.models.gpt_base import DEFAULT_LINE_SEPARATOR, Prompt
from src.models.laps_grammar import LAPSGrammar
from src.models.sample_generator import GPTSampleGenerator
from src.task_loaders import LANGUAGE, PROGRAMS, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LLM_SOLVER]


@ModelRegistry.register
class GPTSolver(GPTSampleGenerator):
    name = "gpt_solver"

    results_file = "gpt_solver_results.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTSolver(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(self, engine=engine)

    def infer_programs_for_tasks(
        self,
        experiment_state,
        task_split: str,
        task_batch_ids: list,
        # Sampling
        n_samples_per_query: int = None,
        n_queries_per_task: int = None,
        n_queries_per_task_base_dsl: int = 0,
        early_stop_on_solution: bool = True,
        max_retries: int = None,
        add_samples: bool = False,
        # Prompt construction
        body_task_types: list = [LANGUAGE, PROGRAMS],
        final_task_types: list = [LANGUAGE],
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        prepend_dsl_description: bool = False,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        # GPT parameters
        temperature: float = 0.40,
        max_tokens_completion_beta: float = 2.0,
        # Resume from prior runs
        resume_strategy: str = None,
        # Utilities
        verbose: bool = False,
    ):
        if (resume_strategy == "first" and experiment_state.is_first_iteration()) or (
            resume_strategy == "every"
        ):
            # If RESUME_CHECKPOINT_DIRECTORY not defined, default to self checkpoint directory
            results_filepath_ext = os.path.join(
                os.getcwd(),
                experiment_state.get_checkpoint_directory_maybe_resume(),
                task_split,
                self.results_file,
            )
            if os.path.exists(results_filepath_ext):
                with open(results_filepath_ext, "r") as f:
                    results_json = json.load(f)

                # Update experiment state from file
                self.add_samples_to_experiment_state(
                    experiment_state=experiment_state,
                    task_split=task_split,
                    parse_results_valid=results_json["parse_results_valid"],
                    evaluate_samples=True,
                    compute_likelihoods=True,
                    add_samples=add_samples,
                )

                # Copy external results file to checkpoint directory
                results_filepath = os.path.join(
                    os.getcwd(),
                    experiment_state.get_checkpoint_directory(),
                    task_split,
                    self.results_file,
                )
                os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
                with open(results_filepath, "w") as f:
                    json.dump(results_json, f, indent=4)

                print(f"Loaded GPT results from: {results_filepath_ext}")
                return {
                    SKIPPED_MODEL_FN: True,
                }
            else:
                print(f"GPT results not found at: {results_filepath_ext}")
                if experiment_state.is_first_iteration() and task_split == TRAIN:
                    raise ValueError("Unable to resume first iteration.")

        task_to_solutions = defaultdict(list)
        results_by_query = []
        parse_results_valid = []

        for task_i, task_id in enumerate(task_batch_ids):
            for query_i in range(n_queries_per_task + n_queries_per_task_base_dsl):

                # After `n_queries_per_task`, run some `n_queries_per_task_base_dsl` with no abstractions
                include_abstractions = query_i < n_queries_per_task

                prompt = self.construct_initial_prompt(
                    experiment_state=experiment_state,
                    task_split=task_split,
                    task_id=task_id,
                    body_task_types=body_task_types,
                    final_task_types=final_task_types,
                    function_name_classes=function_name_classes,
                    include_abstractions=include_abstractions,
                    prepend_dsl_description=prepend_dsl_description,
                    line_separator=line_separator,
                    max_tokens_completion_beta=max_tokens_completion_beta,
                    verbose=False,
                )

                max_retries = len(prompt.body_task_data) or max_retries
                for retry_i in range(max_retries):
                    if retry_i > 0:
                        prompt.remove_last_body_task()
                        print(
                            f"Retry ({retry_i} / {max_retries}): Prompt reduced to {len(prompt)} tasks."
                        )

                    token_stats = self.get_token_stats(
                        prompt=prompt,
                        max_tokens_completion_beta=max_tokens_completion_beta,
                    )

                    completion = self.query_completion(
                        prompt,
                        n_samples=n_samples_per_query,
                        temperature=temperature,
                        max_tokens=token_stats["max_tokens_completion"],
                        line_separator=line_separator,
                    )

                    if not isinstance(completion, OpenAIObject):
                        if isinstance(completion, InvalidRequestError):
                            if retry_i >= max_retries - 1:
                                raise ValueError(f"Max retries {max_retries} exceeded.")
                            continue
                        else:
                            raise ValueError(
                                f"Unexpected completion type: {type(completion)}"
                            )

                    valid_request_types = self.get_valid_request_types(
                        experiment_state, task_split, task_batch_ids
                    )
                    parse_results = self.parse_completion(
                        completion,
                        experiment_state=experiment_state,
                        task_split=task_split,
                        task_ids=[task_id],
                        valid_request_types=valid_request_types,
                        function_name_classes=function_name_classes,
                        evaluate_samples=True,
                        compute_likelihoods=True,
                        verbose=verbose,
                    )
                    results_by_query.append(
                        {
                            "task_id": task_id,
                            "query_i": query_i,
                            "include_abstractions": include_abstractions,
                            "token_stats": token_stats,
                            "prompt": prompt.to_dict(),
                            "completion": completion.to_dict_recursive(),
                            "parse_results": parse_results,
                        }
                    )

                    task_solved = False
                    for result_data in parse_results:
                        result_data["query_i"] = query_i
                        result_data["include_abstractions"] = include_abstractions

                        if result_data["valid"]:
                            parse_results_valid.append(result_data)

                        if result_data.get("tasks_solved"):
                            # Sanity check
                            assert len(result_data["tasks_solved"]) == 1
                            assert result_data["tasks_solved"][0] == task_id
                            task_to_solutions[task_id].append(result_data)
                            task_solved = True

                    # Print query results
                    if verbose:
                        print("-" * 12)
                        print(prompt)
                        print("-" * 12)

                        print(f"GPT ({self.ENGINE}) completions:")
                        for result_data in parse_results:
                            if result_data.get("tasks_solved"):
                                status_emoji = "🏆"
                            elif result_data["valid"]:
                                status_emoji = "❎"
                            else:
                                status_emoji = "❌"
                            print(f"{status_emoji} {result_data['text']}")
                        print("")

                    print(
                        f"[TASK {task_i}/{len(task_batch_ids)} QUERY {query_i}/{n_queries_per_task}]: {task_id}",
                        flush=True,
                    )

                    if (
                        n_queries_per_task_base_dsl > 0
                        and query_i >= n_queries_per_task
                    ):
                        print(
                            f"Queried using Base DSL: {LAPSGrammar.DEFAULT_FUNCTION_NAMES}"
                        )

                    n_tasks_solved = len(
                        [
                            t
                            for t, results in task_to_solutions.items()
                            if len(results) > 0
                        ]
                    )
                    print(
                        f"Tasks solved so far: {n_tasks_solved}/{task_i+1}", flush=True
                    )

                    # Query succeeded: break from retry loop
                    break

                if task_solved and early_stop_on_solution:
                    break

            tasks_solved = [
                t for t, results in task_to_solutions.items() if len(results) > 0
            ]

            # Collect results
            results = {
                "params": {
                    "n_samples_per_query": n_samples_per_query,
                    "n_queries_per_task": n_queries_per_task,
                    "n_queries_per_task_base_dsl": n_queries_per_task_base_dsl,
                    "temperature": temperature,
                    "engine": self.ENGINE,
                    "line_separator": line_separator,
                    "body_task_types": body_task_types,
                    "final_task_types": final_task_types,
                    "function_name_classes": function_name_classes,
                },
                "summary": {
                    "n_tasks_solved": len(tasks_solved),
                    "tasks_solved": list(tasks_solved),
                },
                "task_to_solutions": task_to_solutions,
                "parse_results_valid": parse_results_valid,
                "results_by_query": results_by_query,
            }

            if n_queries_per_task_base_dsl:
                tasks_solved_base_dsl = [
                    t
                    for t, results in task_to_solutions.items()
                    if len(results) > 0
                    and len(list(filter(lambda x: x["include_abstractions"], results)))
                    == 0
                ]
                results["summary"]["n_tasks_solved_base_dsl"] = len(
                    tasks_solved_base_dsl
                )
                results["summary"]["tasks_solved_base_dsl"] = tasks_solved_base_dsl

            # Save results to file
            results_filepath = os.path.join(
                os.getcwd(),
                experiment_state.get_checkpoint_directory(),
                task_split,
                self.results_file,
            )
            os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
            with open(results_filepath, "w") as f:
                json.dump(results, f, indent=4)
            if verbose:
                print(f"Wrote results: {results_filepath}")

        # Update experiment_state
        self.add_samples_to_experiment_state(
            experiment_state=experiment_state,
            task_split=task_split,
            parse_results_valid=parse_results_valid,
            evaluate_samples=True,
            compute_likelihoods=True,
            add_samples=add_samples,
        )

    def construct_initial_prompt(
        self,
        experiment_state,
        task_split,
        task_id,
        body_task_types,
        final_task_types,
        function_name_classes,
        include_abstractions,
        prepend_dsl_description,
        line_separator,
        max_tokens_completion_beta,
        verbose,
    ):
        rng = experiment_state.metadata[RANDOM_GENERATOR]

        non_empty_task_ids = [
            f.task.name
            for f in experiment_state.get_non_empty_frontiers_for_split(TRAIN)
        ]

        # Random ordering of the body tasks
        body_task_ids = list(rng.permutation(non_empty_task_ids))
        if len(body_task_ids) < 2:
            raise ValueError(
                "At least 2 tasks must have non-empty frontiers to construct a prompt."
            )

        prompt = None
        for body_task_i in range(len(body_task_ids)):
            body_task_ids_for_prompt = body_task_ids[: body_task_i + 1]
            prompt_i = Prompt(
                experiment_state=experiment_state,
                body_task_ids=body_task_ids_for_prompt,
                final_task_id=task_id,
                body_task_types=body_task_types,
                final_task_types=final_task_types,
                final_task_split=task_split,
                function_name_classes=function_name_classes,
                include_abstractions=include_abstractions,
                prepend_dsl_description=prepend_dsl_description,
                line_separator=line_separator,
            )

            # Estimate token budgets
            token_stats = self.get_token_stats(
                prompt=prompt_i, max_tokens_completion_beta=max_tokens_completion_beta
            )

            if token_stats["token_count_prompt"] <= token_stats["max_tokens_prompt"]:
                prompt = prompt_i
                if verbose:
                    print(
                        f"Prompt construction ({body_task_i+1} / {len(body_task_ids)}): {token_stats['token_count_prompt']} (prompt; max {token_stats['max_tokens_prompt']}) + {token_stats['max_tokens_completion']} (completion allocation) = {token_stats['token_count_prompt'] + token_stats['max_tokens_completion']} tokens"
                    )
            else:
                break

        if prompt is None:
            raise ValueError(f"Failed to construct prompt.")
        assert body_task_i > 0

        return prompt
