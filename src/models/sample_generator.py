"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import json
import os

import numpy as np
import openai
from openai.error import InvalidRequestError

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import InferenceFailure, ParseFailure, Program
from dreamcoder.task import Task
from src.task_loaders import ALL, TRAIN

openai.api_key = os.environ["OPENAI_API_KEY"]

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(model_loaders.ModelLoader):
    name = "codex_sample_generator"

    query_results_file = "codex_query_results.json"

    ENGINE = "davinci-codex"
    SEP = "\n"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexSampleGenerator(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_samples(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        n_samples: int = 5,
        n_train_programs_per_prompt: int = 10,
        temperature: float = 0.75,
        max_tokens: int = 256,
        debug: bool = False,
    ):
        """
        Queries Codex API to generate new samples.

        params:
            experiment_state: experiment_state
            task_splits: task_splits
            task_ids_in_splits: task_ids_in_splits
            n_samples: Number of programs for Codex to generate
            n_train_programs_per_prompt: Number of training programs to include
                in the Codex prompt.
            debug: If True, replaces live query to Codex with a random sample
                from the training set.

        """
        frontiers = experiment_state.get_frontiers_for_ids(TRAIN, ALL)

        # TODO(gg): Extend to use language
        # language = experiment_state.get_language_for_ids(TRAIN, ALL)

        # Remove frontiers with no programs
        programs_train = [str(e.program) for f in frontiers for e in f.entries]
        if len(programs_train) == 0:
            print("CodexSampleGenerator: No non-empty training frontiers.")
            return None

        # TODO(gg): Prevent generation of duplicate programs
        # programs_train_hashes = set(map(hash, programs_train))

        programs_for_prompt = list(
            np.random.choice(
                programs_train,
                size=min(n_train_programs_per_prompt, len(programs_train)),
                replace=False,
            )
        )
        prompt_text = self.SEP.join(programs_for_prompt) + self.SEP

        print(f"Querying Codex with prompt ({len(programs_for_prompt)} examples)...")
        if debug:
            completion = self.query_mock(experiment_state, n_samples=n_samples)
        else:
            completion = self.query_codex(
                prompt_text,
                n_samples=n_samples,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if completion is not None:
            query_results = {
                "programs_valid": [],
                "programs_invalid": [],
                "prompt_text": prompt_text,
                "prompt_programs": programs_for_prompt,
                "separator": self.SEP,
                "completion": completion.to_dict_recursive(),
            }

            for choice in completion["choices"]:
                program_str = choice["text"]
                try:
                    p = Program.parse(program_str)
                except (ParseFailure, IndexError, AssertionError) as e:
                    print(f"Failed to parse ({type(e)}): {program_str}")
                    query_results["programs_invalid"].append(program_str)
                    continue

                try:
                    p_type = p.infer()
                except InferenceFailure as e:
                    print(e)
                    query_results["programs_invalid"].append(program_str)
                    continue

                query_results["programs_valid"].append(program_str)

                # TODO(gg): avoid adding duplicate generated programs
                # A bit tricky since task-to-program mapping is many-to-many
                program_hash = hash(program_str)

                task = Task(
                    name=f"codex_{program_hash}",
                    request=p_type,
                    examples=[],
                )

                frontier = Frontier(
                    frontier=[
                        FrontierEntry(
                            program=p,
                            logPrior=0.0,
                            logLikelihood=0.0,
                        )
                    ],
                    task=task,
                )

                experiment_state.sample_tasks[TRAIN].append(task)
                experiment_state.sample_frontiers[TRAIN][task] = frontier

            query_results_filepath = os.path.join(
                os.getcwd(),
                experiment_state.get_checkpoint_directory(),
                self.query_results_file,
            )
            with open(query_results_filepath, "w") as f:
                json.dump(query_results, f)
            print(
                f"Codex query results:\nVALID: {len(query_results['programs_valid'])}\nINVALID: {len(query_results['programs_invalid'])}"
            )
            print(f"Wrote results: {query_results_filepath}")
        else:
            # TODO(gg): Better error handling
            print("Query to Codex encountered an error. No samples were added.")

    def query_codex(
        self,
        prompt: str,
        n_samples: int,
        temperature: float = 0.75,
        max_tokens: int = 256,
    ):
        """
        TODO(gg): Make params for temperature, n, and max_tokens
        """
        try:
            completion = openai.Completion.create(
                engine=self.ENGINE,
                prompt=prompt,
                temperature=temperature,
                n=n_samples,
                stop=self.SEP,
                max_tokens=max_tokens,
            )
        except InvalidRequestError as e:
            print(e)
            completion = None

        return completion

    def query_mock(self, experiment_state, n_samples: int = 3, **kwargs):
        """Debugging query that returns a sample of programs from the task."""
        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=TRAIN, task_ids=ALL
        )
        frontiers = np.random.choice(frontiers, size=n_samples)
        program_str_list = [str(e.program) for f in frontiers for e in f.entries]
        completion = dict(choices=[dict(text=p_str) for p_str in program_str_list])
        return completion
