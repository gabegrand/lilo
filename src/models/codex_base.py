"""
codex_base.py | Author: Catherine Wong, Gabe Grand.

Base class containing utilities for working with the Codex language model.
"""

import os

import openai
from openai.error import InvalidRequestError


class CodexBase(object):
    DEFAULT_ENGINE = "davinci-codex"
    DEFAULT_SEPARATOR = "\n"

    def __init__(self, experiment_state=None):
        super().__init__()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
            )
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def query_codex(
        self,
        prompt: str,
        n_samples: int,
        temperature: float = 0.75,
        max_tokens: int = 256,
        engine: str = DEFAULT_ENGINE,
        separator: str = DEFAULT_SEPARATOR,
    ):
        try:
            completion = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                n=n_samples,
                stop=separator,
                max_tokens=max_tokens,
            )
        except InvalidRequestError as e:
            print(e)
            completion = None

        return completion
