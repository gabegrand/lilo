"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import itertools
import json
import os
from re import L


import numpy as np
from openai.api_resources.completion import Completion

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import EtaLongVisitor, InferenceFailure, ParseFailure, Program
from dreamcoder.task import Task
from src.models.codex_base import *
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import ALL, TRAIN, PROGRAMS, LANGUAGE
from src.experiment_iterator import RANDOM_GENERATOR

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(CodexBase, model_loaders.ModelLoader):
    name = "codex_sample_generator"

    query_results_file = "codex_query_results.json"

    PROMPT_EXAMPLE_TYPES = [LANGUAGE, PROGRAMS]

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
        n_samples: int = 5,
        n_samples_per_prompt: int = 1,
        n_train_programs_per_prompt: int = 10,
        temperature: float = 0.75,
        max_tokens: int = 256,
        separator: str = CodexBase.DEFAULT_SEPARATOR,
        language_separator: str = CodexBase.DEFAULT_LANGUAGE_SEPARATOR,
        engine: str = CodexBase.DEFAULT_ENGINE,
        debug: bool = False,
        verbose_prompt: bool = False,
        use_cached: bool = False,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        prompt_example_types: list = [PROGRAMS],
        sample_type: str = PROGRAMS,
        allow_duplicate_examples_per_task: bool = False,
        allow_language_for_disjoint_tasks: bool = True,
    ):
        """
        Queries Codex API to generate new samples based on training data.

        Currently supports only program generation. Generation of language and
        generation of programs conditioned on language are both forthcoming.

        params:
            experiment_state: experiment_state
            n_samples: Total number of sample examples to attempt to generate with Codex.
                        Some of these programs may be invalid; only valid programs are added to the
                        experiment_state.
            n_samples_per_prompt: Number of samples to take from Codex per each generated prompt, which may
                                    contain a random subset of the training examples.
            n_train_programs_per_prompt: Number of training programs to include
                in the Codex prompt. If `n_train_programs_per_prompt` is too high,
                the prompt may exceed the token budget and trigger an `InvalidRequestError`.
            temperature: Codex temperature sampling value in `[0., 1.]` range.
            max_tokens: Max number of tokens for a single program in the completion.
                Codex will stop at `separator` anyway, so this value should be generous.
            engine: Codex `engine` parameter.
            separator: String to insert between examples in the Codex query. Also
                used as the `stop` sequence during generation.
            language_separator: String to insert before language examples in the Codex query.
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
        """
        if task_splits != [TRAIN]:
            raise ValueError(
                f"CodexSampleGenerator expected task_splits=[{TRAIN}], got task_splits={task_splits}"
            )

        query_results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            self.query_results_file,
        )

        # Sample training examples to construct a Codex prompt.
        prompt_examples, prompt_text = self.generate_codex_prompt_text(
            experiment_state,
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            n_train_examples_per_prompt=n_train_programs_per_prompt,
            separator=separator,
            language_separator=language_separator,
            function_name_classes=function_name_classes,
            prompt_example_types=prompt_example_types,
            sample_type=sample_type,
            allow_duplicate_examples_per_task=allow_duplicate_examples_per_task,
            allow_language_for_disjoint_tasks=allow_language_for_disjoint_tasks,
        )

        if len(prompt_examples) == 0:
            print("CodexSampleGenerator: skipping sampling without prompt examples.")
            return None

        print(
            f"Querying Codex with prompt ({len(prompt_examples)} examples) for {n_samples_per_prompt} samples..."
        )
        if verbose_prompt:
            print(prompt_text)
        completion = self.get_completion_for_prompt(
            experiment_state=experiment_state,
            prompt_text=prompt_text,
            query_results_filepath=query_results_filepath,
            n_samples_per_prompt=n_samples_per_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            engine=engine,
            separator=separator,
            use_cached=use_cached,
            debug=debug,
        )

        if completion is not None:
            query_results = {
                "programs_valid": [],
                "programs_invalid": [],
                "prompt_text": prompt_text,
                "prompt_example_types": prompt_example_types,
                "prompt_programs": prompt_examples,
                "engine": engine,
                "separator": separator,
                "completion": completion.to_dict_recursive(),
            }
            programs_valid, programs_invalid = self.maybe_get_frontiers_from_completion(
                experiment_state=experiment_state,
                codex_completion=completion,
                completion_name_classes=function_name_classes,
            )
            query_results["programs_valid"], query_results["programs_invalid"] = (
                list(programs_valid),
                list(programs_invalid),
            )
            if verbose_prompt:
                print(separator.join(programs_invalid))
            if not (debug or use_cached):
                with open(query_results_filepath, "w") as f:
                    json.dump(query_results, f)
                print(f"Wrote results: {query_results_filepath}")
        else:
            raise ValueError("Query to Codex encountered an error.")

    def get_completion_for_prompt(
        self,
        experiment_state,
        prompt_text,
        query_results_filepath,
        n_samples_per_prompt,
        temperature,
        max_tokens,
        engine,
        separator,
        use_cached,
        debug,
    ):
        if debug:
            # Debugging query that returns programs.
            completion = self.query_mock(
                experiment_state, n_samples=n_samples_per_prompt
            )
        elif use_cached and os.path.exists(query_results_filepath):
            print("Using cached examples....")
            # For debugging only - does not verify that the cached completion matches the desired query parameters
            with open(query_results_filepath, "r") as f:
                completion_data = json.load(f)["completion"]
            completion = Completion()
            completion.refresh_from(completion_data)
        else:
            use_cached = False
            completion = self.query_codex(
                prompt_text,
                n_samples=n_samples_per_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine,
                separator=separator,
            )
        return completion

    def maybe_get_frontiers_from_completion(
        self,
        experiment_state,
        codex_completion,
        completion_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
    ):
        """Extract valid sampled programs from a Codex completion.
        :ret: number of frontiers.
        """
        valid_programs, invalid_programs = set(), set()

        grammar = experiment_state.models[model_loaders.GRAMMAR]
        for choice in codex_completion["choices"]:
            program_str_codex = choice["text"]
            try:
                # Write the program back into the DreamCoder form from whatever it was initially in.
                program_str = grammar.show_program(
                    program_str_codex, input_name_class=completion_name_classes
                )
                p = Program.parse(program_str)
            except (ParseFailure, IndexError, AssertionError, ValueError) as e:
                print(f"Failed to parse ({type(e)}): {program_str_codex}")
                invalid_programs.add(program_str_codex)
                continue

            try:
                p_type = p.infer()
            except InferenceFailure:
                print(f"Type inference failure for: {str(p)}")
                invalid_programs.add(program_str_codex)
                continue

            # Hack to avoid fatal error when computing likelihood summaries during rescoreFrontier
            try:
                p = EtaLongVisitor(request=p_type).execute(p)
            except:
                print(f"Error converting to ETA Long for {p}")
                invalid_programs.add(program_str_codex)
                continue

            program_str = str(p)

            valid_programs.add(program_str_codex)

            # NOTE(gg): Hashing for task naming avoids adding duplicate programs to the `experiment_state`
            program_hash = abs(hash(program_str))

            task = Task(name=f"codex_{program_hash}", request=p_type, examples=[],)

            frontier = Frontier(
                frontier=[FrontierEntry(program=p, logPrior=0.0, logLikelihood=0.0,)],
                task=task,
            )

            # Re-score the logPrior and logLikelihood of the frontier under the current grammar
            frontier = experiment_state.models[model_loaders.GRAMMAR].rescoreFrontier(
                frontier
            )

            experiment_state.sample_tasks[TRAIN].append(task)
            experiment_state.sample_frontiers[TRAIN][task] = frontier

        print(
            f"Codex query results:\nVALID: {len(valid_programs)}\nINVALID: {len(invalid_programs)}"
        )
        return valid_programs, invalid_programs

    def sample_prompt_training_examples(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: list,
        n_train_examples_per_prompt: int = 10,
        separator: str = CodexBase.DEFAULT_SEPARATOR,
        language_separator: str = CodexBase.DEFAULT_LANGUAGE_SEPARATOR,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        prompt_example_types: list = [PROGRAMS],
        sample_type: str = PROGRAMS,
        allow_duplicate_examples_per_task: bool = False,
        allow_language_for_disjoint_tasks: bool = True,
    ):
        """
        Samples a list of up to n_train_examples_per_prompt example tuples, consisting of the example information in prompt_example_types.

        These can be concatenated to form a prompt.

        :ret: List of (example) tuples.
        """
        if LANGUAGE in prompt_example_types and sample_type == PROGRAMS:
            # If language, get an extra annotation to precede the program.
            n_train_examples_per_prompt = n_train_examples_per_prompt + 1

        # Sort into the canonical order.
        prompt_example_types = [
            p for p in self.PROMPT_EXAMPLE_TYPES if p in prompt_example_types
        ]

        # Build a list of candidate example tuples.
        candidate_prompt_examples = []
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        rng = experiment_state.metadata[RANDOM_GENERATOR]

        for task_id in task_ids_in_splits[TRAIN]:
            example_types_for_task = []
            for example_type in prompt_example_types:
                # Add any natural language specifications for the task
                if example_type == LANGUAGE:
                    examples_for_task = experiment_state.get_language_for_ids(
                        TRAIN, [task_id]
                    )[0]
                    examples_for_task = [
                        language_separator + e for e in examples_for_task
                    ]
                # Add any programs for the task
                elif example_type == PROGRAMS:
                    frontier = experiment_state.get_frontiers_for_ids(TRAIN, [task_id])[
                        0
                    ]
                    examples_for_task = [e.program for e in frontier.entries]
                    examples_for_task = [
                        grammar.show_program(p, name_classes=function_name_classes)
                        for p in examples_for_task
                    ]
                example_types_for_task.append(examples_for_task)
            if len(example_types_for_task) < 1:
                continue
            # Create cross-product of {spec, program} examples.
            example_tuples = itertools.product(*example_types_for_task)
            if allow_duplicate_examples_per_task:
                candidate_prompt_examples += list(example_tuples)
            else:
                task_example = rng.choice(list(example_tuples))
                candidate_prompt_examples.append(task_example)

        # Sample examples from the candidates
        n_train_examples_per_prompt = min(
            n_train_examples_per_prompt, len(candidate_prompt_examples)
        )
        examples_for_prompt = list(
            rng.choice(
                candidate_prompt_examples,
                size=n_train_examples_per_prompt,
                replace=False,
            )
        )
        examples_for_prompt = [list(e) for e in examples_for_prompt]

        if len(examples_for_prompt) == 0:
            print("CodexSampleGenerator: No training frontiers to sample prompts.")
            return []

        # If prompting with language in order to sample programs from Codex, modify the last example tuple to only include language so we can sample programs as a continuation.
        if prompt_example_types == [LANGUAGE, PROGRAMS] and sample_type == PROGRAMS:
            examples_for_prompt[-1] = [
                examples_for_prompt[-1][0]
            ]  # Don't include the last program
            if allow_language_for_disjoint_tasks:
                final_language_example = self.get_language_for_disjoint_tasks(
                    experiment_state, task_ids_in_splits
                )
                # Randomly select one and randomly select its language
                examples_for_prompt[-1] = [language_separator + final_language_example]

        return examples_for_prompt

    def get_language_for_disjoint_tasks(self, experiment_state, task_ids_in_splits):
        """Samples language for tasks that are NOT in task_ids_in_splits.
        """
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        # Randomly select language from disjoint set of tasks that have non-empty annotations
        disjoint_task_ids = [
            t.name
            for t in experiment_state.tasks[TRAIN]
            if t.name not in task_ids_in_splits[TRAIN]
        ]
        final_language_examples = experiment_state.get_language_for_ids(
            TRAIN, task_ids=disjoint_task_ids
        )
        final_language_examples = [
            rng.choice(l) for l in final_language_examples if len(l) > 0
        ]
        final_language_example = rng.choice(final_language_examples)
        return final_language_example

    def generate_codex_prompt_text(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: list,
        n_train_examples_per_prompt: int = 10,
        separator: str = CodexBase.DEFAULT_SEPARATOR,
        language_separator: str = CodexBase.DEFAULT_LANGUAGE_SEPARATOR,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        prompt_example_types: list = [PROGRAMS],
        sample_type: str = PROGRAMS,
        allow_duplicate_examples_per_task: bool = False,
        allow_language_for_disjoint_tasks: bool = True,
    ):

        training_examples = self.sample_prompt_training_examples(
            experiment_state,
            task_splits,
            task_ids_in_splits,
            n_train_examples_per_prompt,
            separator,
            language_separator,
            function_name_classes,
            prompt_example_types,
            sample_type,
            allow_duplicate_examples_per_task,
            allow_language_for_disjoint_tasks,
        )

        # For now, assume we only want to sample programs.
        prompt_text = (
            separator.join([separator.join(e) for e in training_examples]) + separator
        )

        print(
            f"Constructed prompt containing: {len(training_examples)} {prompt_example_types} examples..."
        )

        # Flatten the training examples so that they're easier to unpack later.
        if len(prompt_example_types) == 1:
            training_examples = [t[0] for t in training_examples]

        return training_examples, prompt_text

    def query_mock(self, experiment_state, n_samples: int = 3, **kwargs):
        """Debugging query that returns a sample of programs from the task."""
        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=ALL
        )
        frontiers = np.random.choice(frontiers, size=n_samples)
        program_str_list = [str(e.program) for f in frontiers for e in f.entries]
        completion = dict(choices=[dict(text=p_str) for p_str in program_str_list])
        return completion
