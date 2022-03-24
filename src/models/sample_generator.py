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
from src.models.codex_base import *
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import ALL, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(CodexBase, model_loaders.ModelLoader):
    name = "codex_sample_generator"

    query_results_file = "codex_query_results.json"

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
        n_train_programs_per_prompt: int = 10,
        temperature: float = 0.75,
        max_tokens: int = 256,
        separator: str = CodexBase.DEFAULT_SEPARATOR,
        engine: str = CodexBase.DEFAULT_ENGINE,
        debug: bool = False,
        use_cached: bool = False,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
    ):
        """
        Queries Codex API to generate new samples based on training data.

        Currently supports only program generation. Generation of language and
        generation of programs conditioned on language are both forthcoming.

        params:
            experiment_state: experiment_state
            n_samples: Number of programs for Codex to generate. Some of these
                programs may be invalid; only valid programs are added to the
                experiment_state.
            n_train_programs_per_prompt: Number of training programs to include
                in the Codex prompt. If `n_train_programs_per_prompt` is too high,
                the prompt may exceed the token budget and trigger an `InvalidRequestError`.
            temperature: Codex temperature sampling value in `[0., 1.]` range.
            max_tokens: Max number of tokens for a single program in the completion.
                Codex will stop at `separator` anyway, so this value should be generous.
            engine: Codex `engine` parameter.
            separator: String to insert between examples in the Codex query. Also
                used as the `stop` sequence during generation.
            debug: If True, replaces live query to Codex with a random sample
                from the training set.
            use_cached: If True, replaces live query to Codex with a cached query
                stored in `query_results_filepath`.
            function_name_classes: An array of 'name_classes' specifying what naming scheme to use for functions
                programs used for the inductive prompt. Name classes will be applied in order as they are avaialble for each
                function, falling back on DEFAULT (the DreamCoder parseable function names).

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

        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=task_ids_in_splits[TRAIN]
        )

        # TODO(gg): Extend to use language
        # language = experiment_state.get_language_for_ids(task_split=TRAIN, task_ids=task_ids_in_splits[TRAIN])

        # Remove frontiers with no programs
        programs_train = [e.program for f in frontiers for e in f.entries]
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        programs_train = [
            grammar.show_program(p, name_classes=function_name_classes)
            for p in programs_train
        ]

        programs_train = [str(p) for p in programs_train]

        if len(programs_train) == 0:
            print("CodexSampleGenerator: No non-empty training frontiers.")
            return None

        # TODO(gg): Prevent generation of duplicate programs
        # programs_train_hashes = set(map(hash, programs_train))
        n_train_programs_per_prompt = min(
            n_train_programs_per_prompt, len(programs_train)
        )

        programs_for_prompt = list(
            np.random.choice(
                programs_train, size=n_train_programs_per_prompt, replace=False,
            )
        )
        prompt_text = separator.join(programs_for_prompt) + separator

        print(f"Querying Codex with prompt ({len(programs_for_prompt)} examples)...")
        if debug:
            completion = self.query_mock(experiment_state, n_samples=n_samples)
        elif use_cached and os.path.exists(query_results_filepath):
            # For debugging only - does not verify that the cached completion matches the desired query parameters
            with open(query_results_filepath, "r") as f:
                completion_data = json.load(f)["completion"]
            completion = Completion()
            completion.refresh_from(completion_data)
        else:
            use_cached = False
            completion = self.query_codex(
                prompt_text,
                n_samples=n_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine,
                separator=separator,
            )

        if completion is not None:
            query_results = {
                "programs_valid": [],
                "programs_invalid": [],
                "prompt_text": prompt_text,
                "prompt_programs": programs_for_prompt,
                "engine": engine,
                "separator": separator,
                "completion": completion.to_dict_recursive(),
            }

            for choice in completion["choices"]:
                program_str_codex = choice["text"]
                try:
                    # Write the program back into the DreamCoder form.
                    program_str = grammar.show_program(
                        program_str_codex, input_name_class=function_name_classes
                    )
                    p = Program.parse(program_str)
                except (ParseFailure, IndexError, AssertionError, ValueError) as e:
                    print(f"Failed to parse ({type(e)}): {program_str_codex}")
                    query_results["programs_invalid"].append(program_str_codex)
                    continue

                try:
                    p_type = p.infer()
                except InferenceFailure:
                    print(f"Type inference failure for: {str(p)}")
                    query_results["programs_invalid"].append(program_str_codex)
                    continue

                # Hack to avoid fatal error when computing likelihood summaries during rescoreFrontier
                try:
                    p = EtaLongVisitor(request=p_type).execute(p)
                except:
                    print(f"Error converting to ETA Long for {p}")
                    query_results["programs_invalid"].append(program_str_codex)
                    continue

                program_str = str(p)

                query_results["programs_valid"].append(program_str_codex)

                # NOTE(gg): Hashing for task naming avoids adding duplicate programs to the `experiment_state`
                program_hash = abs(hash(program_str))

                task = Task(name=f"codex_{program_hash}", request=p_type, examples=[],)

                frontier = Frontier(
                    frontier=[
                        FrontierEntry(program=p, logPrior=0.0, logLikelihood=0.0,)
                    ],
                    task=task,
                )

                # Re-score the logPrior and logLikelihood of the frontier under the current grammar
                frontier = experiment_state.models[
                    model_loaders.GRAMMAR
                ].rescoreFrontier(frontier)

                experiment_state.sample_tasks[TRAIN].append(task)
                experiment_state.sample_frontiers[TRAIN][task] = frontier

            print(
                f"Codex query results:\nVALID: {len(query_results['programs_valid'])}\nINVALID: {len(query_results['programs_invalid'])}"
            )
            if not (debug or use_cached):
                with open(query_results_filepath, "w") as f:
                    json.dump(query_results, f)
                print(f"Wrote results: {query_results_filepath}")
        else:
            raise ValueError("Query to Codex encountered an error.")

    def query_mock(self, experiment_state, n_samples: int = 3, **kwargs):
        """Debugging query that returns a sample of programs from the task."""
        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=ALL
        )
        frontiers = np.random.choice(frontiers, size=n_samples)
        program_str_list = [str(e.program) for f in frontiers for e in f.entries]
        completion = dict(choices=[dict(text=p_str) for p_str in program_str_list])
        return completion
