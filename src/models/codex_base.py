"""
codex_base.py | Author: Catherine Wong, Gabe Grand.

Base class containing utilities for working with the Codex language model.
"""

import json
import os
import time

import openai
from openai.error import InvalidRequestError, RateLimitError

from src.experiment_iterator import RANDOM_GENERATOR
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import GRAMMAR
from src.task_loaders import LANGUAGE, PROGRAMS, TEST, TRAIN

DEFAULT_LINE_SEPARATOR = "\n"


class CodexBase(object):
    DEFAULT_ENGINE = "davinci-codex"

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
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        top_p=None,
        logprobs=None,
        max_attempts_rate_limit=5,
        rate_limit_seconds=30,
    ):
        pause_for_rate_limit = False
        completion = None
        for idx in range(max_attempts_rate_limit):
            if pause_for_rate_limit:
                print(
                    f"ERR: Codex rate limit. On attempt {idx}/{max_attempts_rate_limit} after waiting {rate_limit_seconds}s."
                )
                time.sleep(rate_limit_seconds)
                rate_limit_seconds *= 2  # Exponential backoff
            try:
                completion = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature if top_p is None else 1.0,
                    top_p=top_p if temperature is None else 1.0,
                    n=n_samples,
                    stop=line_separator,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                )
                return completion
            except InvalidRequestError as e:
                print(e)
                return e
            except RateLimitError as e:
                print(e)
                pause_for_rate_limit = True
                completion = e

        return completion


class Prompt(object):
    TASK_TYPES = [LANGUAGE, PROGRAMS]

    DEFAULT_PREFIX_PROGRAM = ""
    DEFAULT_PREFIX_LANGUAGE = "-- "  # Haskell-style comment

    # Final task is the last task in body_tasks
    FINAL_TASK_ORIGIN_DEFAULT = "default"
    # Final task is drawn randomly from unused train tasks
    FINAL_TASK_ORIGIN_RANDOM_TRAIN = "random_train"
    # Final task is drawn randomly from test tasks
    FINAL_TASK_ORIGIN_RANDOM_TEST = "random_test"

    def __init__(
        self,
        experiment_state,
        body_task_ids: list,
        final_task_id: str = None,
        body_task_types: list = [LANGUAGE, PROGRAMS],
        final_task_types: list = [LANGUAGE],
        final_task_origin: str = FINAL_TASK_ORIGIN_DEFAULT,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        prefix_language: str = DEFAULT_PREFIX_LANGUAGE,
        prefix_program: str = DEFAULT_PREFIX_PROGRAM,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
    ):
        # Default final_task_id is the last task in body_task_ids
        if final_task_id is None:
            final_task_id = body_task_ids[-1]
            body_task_ids = body_task_ids[:-1]

        # Enforce canonical ordering of task_types
        body_task_types = [t for t in self.TASK_TYPES if t in body_task_types]
        final_task_types = [t for t in self.TASK_TYPES if t in final_task_types]
        assert len(body_task_types) > 0
        assert len(final_task_types) > 0

        self.experiment_state = experiment_state
        self.grammar = experiment_state.models[GRAMMAR]
        self.rng = experiment_state.metadata[RANDOM_GENERATOR]

        self.body_task_types = body_task_types
        self.final_task_types = final_task_types
        self.final_task_origin = final_task_origin

        self.line_separator = line_separator
        self.prefix_language = prefix_language
        self.prefix_program = prefix_program

        self.function_name_classes = function_name_classes

        self.body_task_data = [
            self._get_task_data(task_id=task_id, task_types=body_task_types)
            for task_id in body_task_ids
        ]
        self.final_task_data = self._get_task_data(
            task_id=final_task_id,
            task_types=final_task_types,
            task_split=TEST
            if final_task_origin is Prompt.FINAL_TASK_ORIGIN_RANDOM_TEST
            else TRAIN,
        )

    def __len__(self):
        return len(self.body_task_data) + 1

    def __repr__(self):
        return self.json()

    def __str__(self):
        prompt_text = ""
        for task_data in self.body_task_data + [self.final_task_data]:
            if task_data["task_language"] is not None:
                prompt_text += (
                    self.prefix_language
                    + task_data["task_language"]
                    + self.line_separator
                )
            if task_data["task_program"] is not None:
                prompt_text += (
                    self.prefix_program
                    + task_data["task_program"]
                    + self.line_separator
                )
        return prompt_text

    def serialize(self):
        return self.__str__()

    def to_dict(self):
        return {
            "body_task_data": self.body_task_data,
            "final_task_data": self.final_task_data,
        }

    def load_from_dict(self, d):
        self.body_task_data = d["body_task_data"]
        self.final_task_data = d["final_task_data"]

    def json(self):
        return json.dumps(self.to_dict(), indent=4)

    def _get_task_data(self, task_id: str, task_types: list, task_split: str = TRAIN):
        frontier = self.experiment_state.get_frontiers_for_ids(task_split, [task_id])[0]
        if PROGRAMS in task_types:
            task_program = self.rng.choice(
                [
                    self.grammar.show_program(
                        e.program, name_classes=self.function_name_classes
                    )
                    for e in frontier.entries
                ]
            )
        else:
            task_program = None
        if LANGUAGE in task_types:
            task_language = self.rng.choice(
                self.experiment_state.get_language_for_ids(task_split, [task_id])[0]
            )
            # Remove any line separators from the language
            task_language = task_language.replace(self.line_separator, " ")
        else:
            task_language = None

        return {
            "task_id": task_id,
            "task_program": task_program,
            "task_language": task_language,
        }
