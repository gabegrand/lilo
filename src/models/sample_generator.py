"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import json
import os

import numpy as np
from openai.api_resources.completion import Completion

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import EtaLongVisitor, InferenceFailure, ParseFailure, Program
from dreamcoder.task import Task
from src.experiment_iterator import RANDOM_GENERATOR
from src.models.codex_base import DEFAULT_LINE_SEPARATOR, CodexBase, Prompt
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import ALL, LANGUAGE, PROGRAMS, TEST, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(CodexBase, model_loaders.ModelLoader):
    name = "codex_sample_generator"

    query_results_file = "codex_query_results.json"

    PROMPT_EXAMPLE_TYPES = [LANGUAGE, PROGRAMS]

    # Final task is the last task in body_tasks
    FINAL_TASK_ORIGIN_DEFAULT = "default"
    # Final task is drawn randomly from unused train tasks
    FINAL_TASK_ORIGIN_RANDOM_TRAIN = "random_train"
    # Final task is drawn randomly from test tasks
    FINAL_TASK_ORIGIN_RANDOM_TEST = "random_test"

    # Parse error codes
    ERROR_PARSE = "parse"
    ERROR_INFER = "infer"
    ERROR_ETA_LONG = "eta_long"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexSampleGenerator(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_samples(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: list,
        # Sampling
        n_samples: int,
        n_samples_per_query: int = None,
        max_queries: int = None,
        # Prompt construction
        n_tasks_per_prompt: int = 10,
        body_task_types: list = [PROGRAMS],
        final_task_types: list = [PROGRAMS],
        final_task_origin: str = FINAL_TASK_ORIGIN_DEFAULT,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        # Codex parameters
        temperature: float = 0.75,
        max_tokens: int = 256,
        engine: str = CodexBase.DEFAULT_ENGINE,
        # Utility
        debug: bool = False,
        use_cached: bool = False,
        query_print_frequency=1,
        compute_likelihoods: bool = False,
    ):
        """
        Queries Codex API to generate new samples based on training data.

        Currently supports only program generation. Generation of language and
        generation of programs conditioned on language are both forthcoming.

        params:
            TODO(gg): Update params docs.
            experiment_state: experiment_state
            n_samples: Total number of sample examples to attempt to generate with Codex.
                        Some of these programs may be invalid; only valid programs are added to the
                        experiment_state.
            n_samples_per_query: Number of samples to take from Codex per each generated prompt, which may
                                    contain a random subset of the training examples. If None, will be set equal to n_samples.
            n_train_programs_per_prompt: Number of training programs to include
                in the Codex prompt. If `n_train_programs_per_prompt` is too high,
                the prompt may exceed the token budget and trigger an `InvalidRequestError`.

            temperature: Codex temperature sampling value in `[0., 1.]` range.
            max_tokens: Max number of tokens for a single program in the completion.
                Codex will stop at `separator` anyway, so this value should be generous.
            engine: Codex `engine` parameter.

            debug: If True, replaces live query to Codex with a random sample
                from the training set.
            use_cached: If True, replaces live query to Codex with a cached query
                stored in `query_results_filepath`.
            function_name_classes: An array of 'name_classes' specifying what naming scheme to use for functions
                programs used for the inductive prompt. Name classes will be applied in order as they are avaialble for each
                function, falling back on DEFAULT (the DreamCoder parseable function names).
            prompt_example_types: An array of example types from {LIBRARY,LANGUAGE, PROGRAMS} that will be included in the prompt for Codex to condition on.
            sample_type: A type in {PROGRAMS, LIBRARY, LANGUAGE} to sample from Codex.
            allow_duplicate_examples_per_task: If True, allow multiple examples for a given task.
            allow_language_for_disjoint_tasks: If True, and including language in the prompt example type, prefix the final sample using language disjoint from what we are otherwise training on.
            compute_likelihoods: Whether to compute log likelihoods of each program
                under the grammar. This requires converting the programs to eta-long form,
                which is error-prone, so we don't do it by default.
        """
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        if task_splits != [TRAIN]:
            raise ValueError(
                f"CodexSampleGenerator expected task_splits=[{TRAIN}], got task_splits={task_splits}"
            )

        query_results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            self.query_results_file,
        )

        # Default to drawing all samples from the same prompt
        if n_samples_per_query is None:
            n_samples_per_query = n_samples

        # Set the number of prompt attempts to something reasonable
        min_queries = np.ceil(n_samples / n_samples_per_query)
        if max_queries is None:
            max_queries = int(2 * min_queries)
        elif max_queries < min_queries:
            raise ValueError(
                f"max_queries={max_queries} must be >= min_queries={min_queries}"
            )

        results_by_query = []
        unique_hashes_valid = set()
        parse_results_valid, parse_results_invalid = [], []

        for query_id in range(max_queries):
            body_task_ids = list(
                rng.choice(
                    task_ids_in_splits[TRAIN], size=n_tasks_per_prompt, replace=False
                )
            )

            if final_task_origin == CodexSampleGenerator.FINAL_TASK_ORIGIN_DEFAULT:
                final_task_id = None
            elif (
                final_task_origin == CodexSampleGenerator.FINAL_TASK_ORIGIN_RANDOM_TRAIN
            ):
                final_task_id = rng.choice(
                    [
                        t.name
                        for t in experiment_state.tasks[TRAIN]
                        if t.name not in task_ids_in_splits[TRAIN]
                    ]
                )
            elif (
                final_task_origin == CodexSampleGenerator.FINAL_TASK_ORIGIN_RANDOM_TEST
            ):
                final_task_id = rng.choice(
                    [t.name for t in experiment_state.tasks[TEST]]
                )
            else:
                raise ValueError(f"Unknown final_task_origin={final_task_origin}")

            prompt = Prompt(
                experiment_state=experiment_state,
                body_task_ids=body_task_ids,
                final_task_id=final_task_id,
                body_task_types=body_task_types,
                final_task_types=final_task_types,
                function_name_classes=function_name_classes,
                line_separator=line_separator,
                # TODO(gg): Support for configuring prompt prefixes.
            )

            if query_id % query_print_frequency == 0:
                print(
                    f"Now on query {query_id}/{max_queries}: with {len(unique_hashes_valid)} / {n_samples} total samples. "
                    f"Querying Codex with prompt ({len(prompt)} tasks) for {n_samples_per_query} samples..."
                )

            completion, cache_used = self.get_completion_for_prompt(
                experiment_state=experiment_state,
                prompt_text=prompt.serialize(),
                query_results_filepath=query_results_filepath,
                n_samples_per_query=n_samples_per_query,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine,
                separator=line_separator,
                use_cached=use_cached,
                debug=debug,
            )

            if completion is not None:
                parse_results = self.parse_completion(
                    completion,
                    grammar,
                    function_name_classes,
                    compute_likelihoods,
                )
                results_by_query.append(
                    {
                        "query_id": query_id,
                        "prompt": prompt.to_dict(),
                        "completion": completion.to_dict_recursive(),
                        "parse_results": parse_results,
                    }
                )
                for result_data in parse_results:
                    result_data["query_id"] = query_id
                    if result_data["valid"]:
                        # Only allow one unique parse per program (even if the original text is different).
                        if result_data["hash"] not in unique_hashes_valid:
                            unique_hashes_valid.add(result_data["hash"])
                            parse_results_valid.append(result_data)
                    else:
                        parse_results_invalid.append(result_data)

                    # Stop as soon as target n_samples is reached, even if there are more valid programs in the results.
                    if len(unique_hashes_valid) >= n_samples:
                        break
            else:
                # TODO(gg): More graceful handling of API query failures.
                raise ValueError("Query to Codex encountered an error.")

        # Save results to file.
        query_results = {
            "params": {
                "n_samples": n_samples,
                "n_samples_per_query": n_samples_per_query,
                "max_queries": max_queries,
                "n_tasks_per_prompt": n_tasks_per_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "engine": engine,
                "line_separator": line_separator,
                "use_cached": use_cached,
                "debug": debug,
                "body_task_types": body_task_types,
                "final_task_types": final_task_types,
                "final_task_origin": final_task_origin,
                "function_name_classes": function_name_classes,
                "compute_likelihoods": compute_likelihoods,
            },
            "results": {
                "n_queries": query_id + 1,
                "n_programs_valid": len(unique_hashes_valid),
                "n_programs_invalid": len(parse_results_invalid),
                "programs_valid": parse_results_valid,
                "programs_invalid": parse_results_invalid,
            },
            "results_by_query": results_by_query,
        }
        if not cache_used:
            with open(query_results_filepath, "w") as f:
                json.dump(query_results, f, indent=4)
            print(f"Wrote results: {query_results_filepath}")

        # Update experiment_state.
        self.add_samples_to_experiment_state(
            experiment_state=experiment_state,
            parse_results_valid=parse_results_valid,
            compute_likelihoods=compute_likelihoods,
        )

    def get_completion_for_prompt(
        self,
        experiment_state,
        prompt_text,
        query_results_filepath,
        n_samples_per_query,
        temperature,
        max_tokens,
        engine,
        separator,
        use_cached,
        debug,
    ):
        if debug:
            # Debugging query that returns programs.
            cache_used = True
            completion = self.query_mock(
                experiment_state, n_samples=n_samples_per_query
            )
        elif use_cached and os.path.exists(query_results_filepath):
            cache_used = True
            print("Using cached examples....")
            # For debugging only - does not verify that the cached completion matches the desired query parameters
            with open(query_results_filepath, "r") as f:
                completion_data = json.load(f)["completion"]
            completion = Completion()
            completion.refresh_from(completion_data)
        else:
            cache_used = False
            completion = self.query_codex(
                prompt_text,
                n_samples=n_samples_per_query,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine,
                separator=separator,
            )
        return completion, cache_used

    def parse_completion(
        self,
        completion,
        grammar,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        compute_likelihoods: bool = False,
    ):
        parse_results = []
        for choice in completion["choices"]:
            program_str_codex = choice["text"]
            # CHECK 1: Does the program parse?
            try:
                # Write the program back into the DreamCoder form from whatever it was initially in.
                program_str = grammar.show_program(
                    program_str_codex, input_name_class=function_name_classes
                )
                p = Program.parse(program_str)
            except (ParseFailure, IndexError, AssertionError, ValueError) as e:
                print(f"Failed to parse ({type(e)}): {program_str_codex}")
                parse_results.append(
                    {
                        "text": program_str_codex,
                        "text_after_show_program": program_str,
                        "valid": False,
                        "error": CodexSampleGenerator.ERROR_PARSE,
                    }
                )
                continue
            # CHECK 2: Does the program typecheck?
            try:
                p_type = p.infer()
            except InferenceFailure:
                print(f"Type inference failure for: {str(p)}")
                import pdb

                pdb.set_trace()
                parse_results.append(
                    {
                        "text": program_str_codex,
                        "text_after_show_program": program_str,
                        "valid": False,
                        "error": CodexSampleGenerator.ERROR_INFER,
                    }
                )
                continue
            # CHECK 3: Can we convert the program to eta long form?
            if compute_likelihoods:
                try:
                    # Hack to avoid fatal error when computing likelihood summaries during rescoreFrontier
                    p = EtaLongVisitor(request=p_type).execute(p)
                except:
                    print(f"Error converting to ETA Long for {p}")
                    parse_results.append(
                        {
                            "text": program_str_codex,
                            "text_after_show_program": program_str,
                            "valid": False,
                            "error": CodexSampleGenerator.ERROR_ETA_LONG,
                        }
                    )
                    continue

            parse_results.append(
                {
                    "text": program_str_codex,
                    "valid": True,
                    "program": str(p),
                    "type": p_type,
                    "hash": abs(hash(str(p))),
                }
            )

        return parse_results

    def add_samples_to_experiment_state(
        self,
        experiment_state,
        parse_results_valid: list,
        compute_likelihoods: bool = False,
    ):
        for result_data in parse_results_valid:
            task = Task(
                name=f"codex_{result_data['hash']}",
                request=result_data["type"],
                examples=[],
            )

            frontier = Frontier(
                frontier=[
                    FrontierEntry(
                        program=Program.parse(result_data["program"]),
                        logPrior=0.0,
                        logLikelihood=0.0,
                    )
                ],
                task=task,
            )

            # Re-score the logPrior and logLikelihood of the frontier under the current grammar
            if compute_likelihoods:
                frontier = experiment_state.models[
                    model_loaders.GRAMMAR
                ].rescoreFrontier(frontier)

            experiment_state.sample_tasks[TRAIN].append(task)
            experiment_state.sample_frontiers[TRAIN][task] = frontier

    def query_mock(self, experiment_state, n_samples: int = 3, **kwargs):
        """Debugging query that returns a sample of programs from the task."""
        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=ALL
        )
        frontiers = np.random.choice(frontiers, size=n_samples)
        program_str_list = [str(e.program) for f in frontiers for e in f.entries]
        completion = dict(choices=[dict(text=p_str) for p_str in program_str_list])
        return completion
