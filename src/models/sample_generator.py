"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import os

import numpy as np
import openai
from openai.error import InvalidRequestError

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import Program
from dreamcoder.task import Task
from src.task_loaders import ALL, TRAIN

openai.api_key = os.environ["OPENAI_API_KEY"]

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(model_loaders.ModelLoader):
    name = "codex_sample_generator"

    ENGINE = "davinci-codex"
    SEP = "\n"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexSampleGenerator(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_samples(
        self, experiment_state, task_splits, task_ids_in_splits, debug=False
    ):
        """
        Queries Codex API to generate new samples.
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
        set(map(hash, programs_train))

        programs_for_prompt = np.random.choice(programs_train, size=5, replace=False)
        prompt = self.SEP.join(programs_for_prompt) + self.SEP

        if debug:
            completion = self.query_mock(experiment_state)
        else:
            completion = self.query_codex(prompt)

        if completion is not None:
            for choice in completion["choices"]:
                program_str = choice["text"]
                try:
                    p = Program.parse(program_str)
                except:
                    print(f"Failed to parse Codex-generated program: {program_str}")
                    continue

                # TODO(gg): avoid adding duplicate generated programs
                # A bit tricky since task-to-program mapping is many-to-many
                program_hash = hash(program_str)

                task = Task(
                    name=f"codex_{program_hash}",
                    request=p.infer(),
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
        else:
            # TODO(gg): Better error handling
            pass

    def query_codex(self, prompt: str, n_samples: int = 3, temperature: int = 0.75):
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
                max_tokens=256,
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
