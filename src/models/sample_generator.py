"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import os

import numpy as np
import openai
from openai.error import InvalidRequestError

import src.models.model_loaders as model_loaders
from dreamcoder.program import Program
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

    def generate_samples(self, experiment_state, task_splits, task_ids_in_splits):
        """
        Queries Codex API to generate new samples.
        """
        frontiers = experiment_state.get_frontiers_for_ids(TRAIN, ALL)
        # language = experiment_state.get_language_for_ids(TRAIN, ALL)

        # Remove frontiers with no programs
        programs = [str(e.program) for f in frontiers for e in f.entries]
        if len(programs) == 0:
            print("CodexSampleGenerator: No programs to sample.")
            return None

        programs = np.random.choice(programs, size=5, replace=False)

        prompt = self.SEP.join(programs) + self.SEP

        completion = self.query_codex(prompt)

        # TODO(gg): Better error handling
        if completion is not None:
            sampled_programs = []
            for choice in completion.choices:
                p = choice.text
                try:
                    Program.parse(p)
                    sampled_programs.append(p)
                except:
                    pass

        # TODO(gg): Add sampled_programs to experiment_state

        import pdb

        pdb.set_trace()
        quit()

    def query_codex(self, prompt: str, temperature: int = 0.75):
        """
        TODO(gg): Make params for temperature, n, and max_tokens
        """
        try:
            completion = openai.Completion.create(
                engine=self.ENGINE,
                prompt=prompt,
                temperature=temperature,
                n=3,
                stop=self.SEP,
                max_tokens=256,
            )
        except InvalidRequestError:
            completion = None

        return completion
