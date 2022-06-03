"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import json
import os
from typing import Set

import numpy as np
from openai.api_resources.completion import Completion
from openai.error import APIConnectionError, InvalidRequestError, RateLimitError
from openai.openai_object import OpenAIObject

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import EtaLongVisitor, InferenceFailure, ParseFailure, Program
from dreamcoder.task import Task
from dreamcoder.type import Type, TypeConstructor
from src.experiment_iterator import RANDOM_GENERATOR
from src.models.codex_base import DEFAULT_LINE_SEPARATOR, CodexBase, Prompt
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import ALL, PROGRAMS, TEST, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(CodexBase, model_loaders.ModelLoader):
    name = "codex_sample_generator"

    query_results_file = "codex_query_results.json"

    # Parse error codes
    ERROR_PARSE = "parse"
    ERROR_INFER = "infer"
    ERROR_INVALID_TYPE = "invalid_type"
    ERROR_FREE_VARIABLES = "free_variables"
    ERROR_ETA_LONG = "eta_long"

    # Final task is the last task in body_tasks
    FINAL_TASK_ORIGIN_DEFAULT = "default"
    # Final task is drawn randomly from unused train tasks
    FINAL_TASK_ORIGIN_RANDOM_TRAIN = "random_train"
    # Final task is drawn randomly from test tasks
    FINAL_TASK_ORIGIN_RANDOM_TEST = "random_test"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexSampleGenerator(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_samples(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: dict,
        # Sampling
        n_samples: int,
        n_samples_per_query: int = None,
        max_queries: int = None,
        max_retries: int = None,
        # Prompt construction
        body_task_types: list = [PROGRAMS],
        final_task_types: list = [PROGRAMS],
        final_task_origin: str = FINAL_TASK_ORIGIN_DEFAULT,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        # Codex parameters
        temperature: float = 0.75,
        max_tokens_completion_beta: float = 2.0,
        engine: str = CodexBase.DEFAULT_ENGINE,
        # Utility
        debug: bool = False,
        use_cached: bool = False,
        query_print_frequency: int = 1,
        compute_likelihoods: bool = True,
        verbose: bool = False,
    ):
        """
        Queries Codex API to generate new samples based on training data.

        params:
            # LAPS parameters
            experiment_state: experiment_state
            task_splits: list of task splits
            task_ids_in_splits: dict of task_ids_in_splits

            # Sampling parameters
            n_samples: Total number of unique, valid samples to generate from Codex.
                Prompting will continue until this number is reached or max_queries is exceeded.
            n_samples_per_query: Number of samples to take from Codex per query. Each query uses a new, random prompt.
                Defaults to a single query with n_samples.
            max_queries: Maximum number of queries to make to Codex. Defaults to 2 * min_queries, where min_queries is
                the minimum number of queries required to generate n_samples.
            max_retries: Max number of retries per query.
                Intention is to more gracefully handle `InvalidRequestError` when max tokens is exceeded via iterative backoff.
                Iteratively removes last item from body_tasks until query success or max_retries is exceeded.
                Defaults to a very permissive behavior where the query will retry until reduced to a single task before failing.

            # Prompt construction parameters
            body_task_types: List of task types in [LANGUAGE, PROGRAMS] to include in the body of the prompt.
            final_task_types: List of task types in [LANGUAGE, PROGRAMS] to include in the final task of the prompt.
            final_task_origin: Origin of the final task in the prompt.
            function_name_classes: List of 'name_classes' specifying what naming scheme to use for functions
                programs used for the inductive prompt. Name classes will be applied in order as they are avaialble for each
                function, falling back on DEFAULT (the DreamCoder parseable function names).

            # Codex-specific parameters
            temperature: Codex temperature sampling value in `[0., 1.]` range.
            max_tokens_completion_beta: Multiplicative factor for the maximum number of tokens in the completion.
                max_tokens is set to the number of tokens in the last program in the prompt,
                times the value of max_tokens_completion_beta.
            engine: Codex `engine` parameter.

            # Utility parameters
            debug: If True, replaces live query to Codex with a random sample
                from the training set.
            use_cached: If True, replaces live query to Codex with a cached query
                stored in `query_results_filepath`.
            query_print_frequency: Number of queries to make before printing a status update.
            compute_likelihoods: If True, compute likelihoods for each sample.
            verbose: If True, print extra status updates including parse errors.
        """
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        if task_splits != [TRAIN]:
            raise ValueError(
                f"CodexSampleGenerator expected task_splits=[{TRAIN}], got task_splits={task_splits}"
            )

        # Codex-generated programs must type-infer to a request type in this set
        train_task_request_types = set(
            [
                t.request
                for t in experiment_state.get_tasks_for_ids(
                    TRAIN, task_ids_in_splits[TRAIN]
                )
            ]
        )

        train_programs = set()
        for f in experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=task_ids_in_splits[TRAIN]
        ):
            train_programs.update([e.program for e in f.entries])

        query_results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            self.query_results_file,
        )
        if use_cached and not os.path.exists(query_results_filepath):
            print(
                f"WARNING: No query results found at {query_results_filepath}. Disabling use_cached."
            )
            use_cached = False

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
        sampled_programs = set()
        parse_results_valid = []

        for query_id in range(max_queries):
            # Construct an initial prompt with the max number of tasks we think
            # we can fit based on estimates from GPT-2 tokenizer.
            prompt = self.construct_initial_prompt(
                experiment_state=experiment_state,
                task_ids_in_splits=task_ids_in_splits,
                body_task_types=body_task_types,
                final_task_types=final_task_types,
                final_task_origin=final_task_origin,
                function_name_classes=function_name_classes,
                line_separator=line_separator,
                max_tokens_completion_beta=max_tokens_completion_beta,
                verbose=verbose,
            )

            # Iteratively remove tasks from the prompt until query success.
            max_retries = len(prompt.body_task_data)
            for retry_i in range(max_retries):
                if retry_i > 0:
                    prompt.remove_last_body_task()
                    print(
                        f"Retry ({retry_i} / {max_retries}): Prompt reduced to {len(prompt)} tasks."
                    )

                token_stats = self.get_token_stats(
                    prompt=prompt, max_tokens_completion_beta=max_tokens_completion_beta
                )

                if use_cached:
                    # Load cached prompt for query_id
                    with open(query_results_filepath, "r") as f:
                        prompt_json = json.load(f)["results_by_query"][query_id][
                            "prompt"
                        ]
                    prompt.load_from_dict(prompt_json)

                if query_id % query_print_frequency == 0:
                    print(
                        f"[QUERY {query_id}/{max_queries}]: Prompting Codex ({len(prompt)} tasks, {token_stats['token_count_prompt']} tokens) for {n_samples_per_query} samples ({token_stats['max_tokens_completion']} max tokens)..."
                    )

                completion, cache_used = self.get_completion_for_prompt(
                    query_id=query_id,
                    experiment_state=experiment_state,
                    prompt_text=prompt.serialize(),
                    query_results_filepath=query_results_filepath,
                    n_samples_per_query=n_samples_per_query,
                    temperature=temperature,
                    max_tokens=token_stats["max_tokens_completion"],
                    engine=engine,
                    line_separator=line_separator,
                    use_cached=use_cached,
                    debug=debug,
                )

                if not isinstance(completion, OpenAIObject):
                    if isinstance(completion, InvalidRequestError):
                        if retry_i >= max_retries - 1:
                            raise ValueError(f"Max retries {max_retries} exceeded.")
                        continue
                    elif isinstance(completion, RateLimitError):
                        raise completion
                    elif isinstance(completion, APIConnectionError):
                        raise completion
                    elif isinstance(completion, dict):
                        # completion is a dict when debug=True
                        assert debug
                    else:
                        raise ValueError(
                            f"Unexpected completion type: {type(completion)}"
                        )

                parse_results = self.parse_completion(
                    completion,
                    grammar,
                    valid_request_types=train_task_request_types,
                    function_name_classes=function_name_classes,
                    compute_likelihoods=compute_likelihoods,
                    verbose=verbose,
                )
                results_by_query.append(
                    {
                        "query_id": query_id,
                        "token_stats": token_stats,
                        "prompt": prompt.to_dict(),
                        "completion": completion.to_dict_recursive()
                        if not debug
                        else completion,
                        "parse_results": parse_results,
                    }
                )
                for result_data in parse_results:
                    result_data["query_id"] = query_id
                    if result_data["valid"]:
                        p = Program.parse(result_data["program"])
                        if (p not in train_programs) and (p not in sampled_programs):
                            sampled_programs.add(p)
                            parse_results_valid.append(result_data)

                    # Stop as soon as target n_samples is reached, even if there are more valid programs in the results.
                    if len(sampled_programs) >= n_samples:
                        break

                if query_id % query_print_frequency == 0:
                    print(
                        f"[QUERY {query_id}/{max_queries}]: Returned {len(list(filter(lambda x: x['valid'], parse_results)))}/{n_samples_per_query} valid samples."
                    )

                print(
                    f"[STATUS]: Sampled {len(sampled_programs)}/{n_samples} unique, valid samples."
                )

                break

            if len(sampled_programs) >= n_samples:
                break

        # Save results to file.
        query_results = {
            "params": {
                "n_samples": n_samples,
                "n_samples_per_query": n_samples_per_query,
                "max_queries": max_queries,
                "temperature": temperature,
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
                "n_sampled_programs": len(sampled_programs),
                "programs_valid": parse_results_valid,
            },
            "results_by_query": results_by_query,
        }
        if not debug and not cache_used:
            with open(query_results_filepath, "w") as f:
                json.dump(query_results, f, indent=4)
            print(f"Wrote results: {query_results_filepath}")

        # Update experiment_state.
        self.add_samples_to_experiment_state(
            experiment_state=experiment_state,
            parse_results_valid=parse_results_valid,
            compute_likelihoods=compute_likelihoods,
        )

        return query_results

    def construct_initial_prompt(
        self,
        experiment_state,
        task_ids_in_splits,
        body_task_types,
        final_task_types,
        final_task_origin,
        function_name_classes,
        line_separator,
        max_tokens_completion_beta,
        verbose,
    ):
        rng = experiment_state.metadata[RANDOM_GENERATOR]

        # Random ordering of the body tasks
        body_task_ids = list(rng.permutation(task_ids_in_splits[TRAIN]))

        if final_task_origin == CodexSampleGenerator.FINAL_TASK_ORIGIN_DEFAULT:
            final_task_id = body_task_ids[-1]
            body_task_ids = body_task_ids[:-1]
        elif final_task_origin == CodexSampleGenerator.FINAL_TASK_ORIGIN_RANDOM_TRAIN:
            final_task_id = rng.choice(
                [
                    t.name
                    for t in experiment_state.tasks[TRAIN]
                    if t.name not in task_ids_in_splits[TRAIN]
                ]
            )
        elif final_task_origin == CodexSampleGenerator.FINAL_TASK_ORIGIN_RANDOM_TEST:
            final_task_id = rng.choice([t.name for t in experiment_state.tasks[TEST]])
        else:
            raise ValueError(f"Unknown final_task_origin={final_task_origin}")

        # Iteratively add tasks to the body until we exceed the token budget
        prompt = None
        for body_task_i in range(len(body_task_ids)):
            body_task_ids_for_prompt = body_task_ids[: body_task_i + 1]
            prompt_i = Prompt(
                experiment_state=experiment_state,
                body_task_ids=body_task_ids_for_prompt,
                final_task_id=final_task_id,
                body_task_types=body_task_types,
                final_task_types=final_task_types,
                final_task_split=(
                    TEST
                    if final_task_origin
                    == CodexSampleGenerator.FINAL_TASK_ORIGIN_RANDOM_TEST
                    else TRAIN
                ),
                function_name_classes=function_name_classes,
                line_separator=line_separator,
                # TODO(gg): Support for configuring prompt prefixes.
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

    def get_token_stats(self, prompt, max_tokens_completion_beta):
        token_count_last_program = self.count_tokens_gpt2(
            str(prompt.get_last_program())
        )
        token_count_prompt = self.count_tokens_gpt2(prompt.serialize())

        # Allocate some multiple of the last program's tokens for the completion
        max_tokens_completion = int(
            token_count_last_program * max_tokens_completion_beta
        )
        # Allocate the remainder of the token budget to the prompt
        max_tokens_prompt = int(self.ENGINE_MAX_TOKENS - max_tokens_completion)

        token_stats = {
            "token_count_prompt": token_count_prompt,
            "token_count_last_program": token_count_last_program,
            "max_tokens_prompt": max_tokens_prompt,
            "max_tokens_completion": max_tokens_completion,
        }
        return token_stats

    def get_completion_for_prompt(
        self,
        query_id,
        experiment_state,
        prompt_text,
        query_results_filepath,
        n_samples_per_query,
        temperature,
        max_tokens,
        engine,
        line_separator,
        use_cached,
        debug,
    ):
        if debug:
            # Debugging query that returns programs.
            cache_used = True
            completion = self.query_mock(
                experiment_state, n_samples=n_samples_per_query
            )
        # For debugging only - does not verify that the cached completion matches the desired query parameters
        elif use_cached and os.path.exists(query_results_filepath):
            cache_used = True
            print("Using cached examples....")
            with open(query_results_filepath, "r") as f:
                query_results = json.load(f)
                # Ensure that the cached query matches the desired query parameters.
                assert (
                    query_results["params"]["n_samples_per_query"]
                    == n_samples_per_query
                )
                assert query_results["params"]["temperature"] == temperature
                assert query_results["params"]["engine"] == engine
                assert query_results["params"]["line_separator"] == line_separator
                # Get the cached completion for the particular query_id.
                assert (
                    query_results["results_by_query"][query_id]["query_id"] == query_id
                )
                completion_data = query_results["results_by_query"][query_id][
                    "completion"
                ]
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
                line_separator=line_separator,
            )
        return completion, cache_used

    def parse_completion(
        self,
        completion,
        grammar,
        valid_request_types: Set[TypeConstructor] = None,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        compute_likelihoods: bool = True,
        verbose: bool = False,
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
            except (
                ParseFailure,
                IndexError,
                AssertionError,
                ValueError,
                AttributeError,
            ) as e:
                if verbose:
                    print(f"Failed to parse ({type(e)}): {program_str_codex}")
                parse_results.append(
                    {
                        "text": program_str_codex,
                        "valid": False,
                        "error": CodexSampleGenerator.ERROR_PARSE,
                    }
                )
                continue
            # CHECK 2: Does the program typecheck?
            try:
                p_type = p.infer()
            except InferenceFailure:
                if verbose:
                    print(f"Type inference failure for: {str(p)}")
                parse_results.append(
                    {
                        "text": program_str_codex,
                        "valid": False,
                        "error": CodexSampleGenerator.ERROR_INFER,
                    }
                )
                continue
            # CHECK 3: Is the inferred type in the set of valid request types?
            if valid_request_types is not None:
                if p_type not in valid_request_types:
                    if verbose:
                        print(
                            f"Inferred type {str(p_type)} not in `valid_request_types` {valid_request_types} for program: {str(p)}"
                        )
                    parse_results.append(
                        {
                            "text": program_str_codex,
                            "valid": False,
                            "error": CodexSampleGenerator.ERROR_INVALID_TYPE,
                        }
                    )
                    continue
            # CHECK 4: Does the program have free variables?
            if not p.closed:
                if verbose:
                    print(f"Program has free variables: {str(p)}")
                parse_results.append(
                    {
                        "text": program_str_codex,
                        "valid": False,
                        "error": CodexSampleGenerator.ERROR_FREE_VARIABLES,
                    }
                )
                continue
            # CHECK 5: Can we convert the program to eta long form?
            if compute_likelihoods:
                try:
                    # Hack to avoid fatal error when computing likelihood summaries during rescoreFrontier
                    p = EtaLongVisitor(request=p_type).execute(p)
                except:
                    if verbose:
                        print(f"Error converting to ETA Long for {p}")
                    parse_results.append(
                        {
                            "text": program_str_codex,
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
                    "type": str(p_type),
                    "type_json": p_type.json(),
                    "hash": abs(hash(str(p))),
                }
            )

        return parse_results

    def add_samples_to_experiment_state(
        self,
        experiment_state,
        parse_results_valid: list,
        compute_likelihoods: bool = True,
    ):
        for result_data in parse_results_valid:
            task = Task(
                name=f"codex_{result_data['hash']}",
                request=Type.fromjson(result_data["type_json"]),
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
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=ALL
        )
        frontiers = rng.choice(frontiers, size=n_samples, replace=True)
        program_str_list = [str(f.entries[0].program) for f in frontiers]
        completion = dict(choices=[dict(text=p_str) for p_str in program_str_list])
        return completion
