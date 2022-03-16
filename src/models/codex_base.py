"""
codex_base.py | Author: Catherine Wong, Gabe Grand.

Base class containing utilities for working with the Codex language model.
"""

import os
import time
import openai
from openai.error import InvalidRequestError, RateLimitError

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
    )
openai.api_key = os.environ["OPENAI_API_KEY"]


class CodexBase(object):
    DEFAULT_ENGINE = "davinci-codex"
    DEFAULT_SEPARATOR = "\n"

    def __init__(self, experiment_state=None):
        super().__init__()

    def query_codex(
        self,
        prompt: str,
        n_samples: int,
        temperature: float = 0.75,
        max_tokens: int = 256,
        engine: str = DEFAULT_ENGINE,
        separator: str = DEFAULT_SEPARATOR,
        top_p=None,
        logprobs=None,
        max_attempts_rate_limit=2,
        rate_limit_seconds=60,
    ):
        pause_for_rate_limit = False
        for idx in range(max_attempts_rate_limit):
            if pause_for_rate_limit:
                print(
                    f"ERR: Codex rate limit. On attempt {idx}/{max_attempts_rate_limit} after waiting {rate_limit_seconds}s."
                )
                time.sleep(rate_limit_seconds)
            try:
                completion = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature if top_p is None else 1.0,
                    top_p=top_p if temperature is None else 1.0,
                    n=n_samples,
                    stop=separator,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                )
                return completion
            except InvalidRequestError as e:
                print(e)
                completion = None
                return completion
            except RateLimitError as e:
                print(e)
                pause_for_rate_limit = True
        return completion
