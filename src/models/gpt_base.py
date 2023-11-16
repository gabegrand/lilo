"""
gpt_base.py | Author: Gabe Grand.

Base class containing utilities for working with the GPT language model.
"""

import json
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Union

import openai
from openai.error import (
    APIConnectionError,
    APIError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
from transformers import GPT2TokenizerFast

from src.experiment_iterator import RANDOM_GENERATOR
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import GRAMMAR
from src.task_loaders import LANGUAGE, PROGRAMS, TEST, TRAIN

DEFAULT_LINE_SEPARATOR = "\n"


class BasePrompt(metaclass=ABCMeta):
    TASK_TYPES = [LANGUAGE, PROGRAMS]

    DEFAULT_MESSAGE_SEPARATOR = (
        DEFAULT_LINE_SEPARATOR + "======" + DEFAULT_LINE_SEPARATOR
    )

    DEFAULT_PREFIX_PROGRAM = ""
    DEFAULT_PREFIX_LANGUAGE = "-- "  # Haskell-style comment

    # https://platform.openai.com/docs/api-reference/chat
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def load_from_dict(self):
        pass

    @abstractmethod
    def to_chat_format(self):
        pass

    def __repr__(self):
        return self.json()

    def json(self):
        return json.dumps(self.to_dict(), indent=4)

    def serialize(self):
        return self.__str__()

    def chat_message(self, text, role=None):
        role = role or self.ROLE_USER
        return {
            "role": role,
            "content": text,
        }


class Prompt(BasePrompt):
    def __init__(
        self,
        experiment_state,
        body_task_ids: list,
        final_task_id: str,
        body_task_types: list = [LANGUAGE, PROGRAMS],
        final_task_types: list = [LANGUAGE],
        final_task_split: str = TRAIN,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        prefix_language: str = BasePrompt.DEFAULT_PREFIX_LANGUAGE,
        prefix_program: str = BasePrompt.DEFAULT_PREFIX_PROGRAM,
        function_name_classes: list = [
            LAPSGrammar.HUMAN_READABLE,
            LAPSGrammar.DEFAULT_FUNCTION_NAMES,
        ],
        include_abstractions: bool = True,
        prepend_dsl_description: bool = False,
    ):
        assert isinstance(body_task_ids, list)
        assert len(body_task_ids) > 0

        assert isinstance(final_task_id, str)
        assert final_task_split in (TRAIN, TEST)

        # Enforce canonical ordering of task_types
        body_task_types = [t for t in self.TASK_TYPES if t in body_task_types]
        final_task_types = [t for t in self.TASK_TYPES if t in final_task_types]
        assert len(body_task_types) > 0
        assert len(final_task_types) > 0
        assert PROGRAMS in body_task_types

        self.experiment_state = experiment_state
        self.grammar = experiment_state.models[GRAMMAR]
        self.rng = experiment_state.metadata[RANDOM_GENERATOR]

        self.body_task_types = body_task_types
        self.final_task_types = final_task_types
        self.final_task_split = final_task_split

        self.line_separator = line_separator
        self.prefix_language = prefix_language
        self.prefix_program = prefix_program

        self.function_name_classes = function_name_classes

        self.body_task_data = [
            self._get_task_data(
                task_split=TRAIN,
                task_id=task_id,
                task_types=body_task_types,
                beta_reduce_program=(not include_abstractions),
            )
            for task_id in body_task_ids
        ]

        self.final_task_data = self._get_task_data(
            task_id=final_task_id,
            task_types=final_task_types,
            task_split=final_task_split,
            beta_reduce_program=(not include_abstractions),
        )

        self.prepend_dsl_description = prepend_dsl_description
        self.dsl_description = (
            self._get_dsl_description(include_abstractions=include_abstractions)
            if prepend_dsl_description
            else ""
        )

    def __len__(self):
        return len(self.body_task_data) + 1

    def __str__(self):
        return (
            self.line_separator.join([x["content"] for x in self.to_message_list()])
            + "\n"
        )

    def to_message_list(self):
        prompt_list = []
        if self.prepend_dsl_description:
            prompt_list += [
                self.chat_message(self.dsl_description, role=self.ROLE_SYSTEM)
            ]
        # Write the body tasks
        prompt_list += [self.chat_message("Here are some example programs:")]
        for task_data in self.body_task_data:
            if LANGUAGE in self.body_task_types:
                prompt_list += [
                    self.chat_message(self.prefix_language + task_data["task_language"])
                ]
            if PROGRAMS in self.body_task_types:
                prompt_list += [
                    self.chat_message(
                        self.prefix_program + task_data["task_program"],
                        role=self.ROLE_ASSISTANT,
                    )
                ]
        # Write the final task
        if LANGUAGE in self.final_task_types:
            prompt_list += [
                self.chat_message(
                    self.prefix_language + self.final_task_data["task_language"],
                )
            ]
        if PROGRAMS in self.final_task_types:
            prompt_list += [
                self.chat_message(
                    self.prefix_program + self.final_task_data["task_program"],
                    role=self.ROLE_ASSISTANT,
                )
            ]
        return prompt_list

    def to_chat_format(self):
        messages = self.to_message_list()
        return messages

    def to_dict(self):
        return {
            "dsl_description": self.dsl_description,
            "body_task_data": self.body_task_data,
            "final_task_data": self.final_task_data,
        }

    def load_from_dict(self, d):
        self.dsl_description = d["dsl_description"]
        self.body_task_data = d["body_task_data"]
        self.final_task_data = d["final_task_data"]

    def get_last_program(self):
        if PROGRAMS in self.final_task_types:
            return self.final_task_data["task_program"]
        else:
            return self.body_task_data[-1]["task_program"]

    def remove_last_body_task(self):
        if len(self.body_task_data) > 1:
            self.body_task_data = self.body_task_data[:-1]
        else:
            raise ValueError("Cannot remove single remaining body task from prompt.")

    def _get_task_data(
        self,
        task_id: str,
        task_types: list,
        task_split: str = TRAIN,
        use_mdl_program: bool = True,
        beta_reduce_program: bool = False,
    ):
        frontier = self.experiment_state.get_frontiers_for_ids(task_split, [task_id])[0]

        # Optionally, get the program
        if PROGRAMS in task_types:
            programs = [e.program for e in frontier.entries]
            if use_mdl_program:
                task_program = self.rng.choice(self.grammar.get_mdl_programs(programs))
            else:
                task_program = self.rng.choice(programs)
            if beta_reduce_program:
                task_program = task_program.betaNormalForm()
            task_program = self.grammar.show_program(
                task_program, name_classes=self.function_name_classes
            )
        else:
            task_program = None

        # Optionally, get the language
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

    def _get_dsl_description(self, include_abstractions: bool = True):
        dsl_fns = []
        for primitive in self.grammar.primitives:
            if primitive.isInvented and (not include_abstractions):
                # Optionally, skip abstractions
                continue
            fn_name = self.grammar.get_name(
                production_key=str(primitive), name_classes=self.function_name_classes
            )
            fn_type = primitive.infer()
            if primitive.isInvented:
                fn_body = str(
                    self.grammar.show_program(
                        str(primitive)[
                            1:
                        ],  # Remove leading `#` so that any inlined abstractions are replaced with their fn_name
                        name_classes=[
                            LAPSGrammar.HUMAN_READABLE,
                            LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                        ],
                    )
                )
            else:
                fn_body = str(primitive)
            fn_description = self.grammar.get_function_description(str(primitive))
            dsl_fns.append((primitive, fn_name, fn_type, fn_body, fn_description))

        dsl_description = (
            "You are an expert programmer working in a language based on lambda calculus.\n"
            + "Your goal is to write programs that accomplish the tasks specified by the user.\n"
        )
        if "dsl_description_prefix" in self.experiment_state.metadata:
            dsl_description += (
                self.experiment_state.metadata["dsl_description_prefix"] + "\n"
            )

        dsl_description += "\nWrite programs using the available functions:\n\n"

        for primitive, fn_name, fn_type, fn_body, fn_description in dsl_fns:
            docstring = f"{fn_name} :: {fn_type}"
            if primitive.isInvented:
                docstring += f"\n{fn_body}"
            if fn_description is not None:
                docstring += f"\ndescription: {fn_description}"
            dsl_description += docstring + "\n\n"

        return dsl_description


class GPTBase(object):
    # https://platform.openai.com/docs/models
    ENGINE_CODEX = "code-davinci-002"
    ENGINE_GPT_3_5_TURBO = "gpt-3.5-turbo"
    ENGINE_GPT_3_5_TURBO_0301 = "gpt-3.5-turbo-0301"
    ENGINE_GPT_4 = "gpt-4"
    ENGINE_GPT_4_0314 = "gpt-4-0314"
    ENGINE_DEFAULT = ENGINE_CODEX

    # Max tokens for BOTH the prompt and the completion.
    MAX_TOKENS_PER_ENGINE = {
        ENGINE_CODEX: 4096,  # 8001
        ENGINE_GPT_3_5_TURBO: 4096,
        ENGINE_GPT_3_5_TURBO_0301: 4096,
        ENGINE_GPT_4: 8192,
        ENGINE_GPT_4_0314: 8192,
    }

    # Models that use chat completion format
    CHAT_ENGINES = [
        ENGINE_GPT_3_5_TURBO,
        ENGINE_GPT_3_5_TURBO_0301,
        ENGINE_GPT_4,
        ENGINE_GPT_4_0314,
    ]

    def __init__(self, experiment_state=None, engine=None):
        super().__init__()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
            )
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.ENGINE = engine or self.ENGINE_DEFAULT
        self.ENGINE_MAX_TOKENS = self.MAX_TOKENS_PER_ENGINE[self.ENGINE]

        # Used for computing approximate token counts for queries
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = self.ENGINE_MAX_TOKENS
        os.environ["TOKENIZERS_PARALLELISM"] = str(False)

    def query_completion(
        self,
        prompt: Union[Prompt, str],
        n_samples: int,
        best_of: int = 1,
        temperature: float = None,
        max_tokens: int = 256,  # Max tokens for completion only.
        stop: str = DEFAULT_LINE_SEPARATOR,
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
                    f"ERR: OpenAI rate limit. On attempt {idx}/{max_attempts_rate_limit} after waiting {rate_limit_seconds}s."
                )
                time.sleep(rate_limit_seconds)
                rate_limit_seconds *= 2  # Exponential backoff
            try:
                completion = self._create_completion(
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n_samples=n_samples,
                    stop=stop,
                    best_of=best_of,
                    line_separator=line_separator,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                )
                return completion
            except InvalidRequestError as e:
                print(e)
                return e
            except (
                RateLimitError,
                APIConnectionError,
                APIError,
                ServiceUnavailableError,
            ) as e:
                print(e)
                pause_for_rate_limit = True
                completion = e

        return completion

    def is_chat_format(self):
        return self.ENGINE in self.CHAT_ENGINES

    def _create_completion(
        self,
        prompt,
        temperature,
        top_p,
        n_samples,
        best_of,
        stop,
        line_separator,
        max_tokens,
        logprobs,
    ):
        if self.is_chat_format():

            # Convert prompt text to ChatCompletion format
            if isinstance(prompt, BasePrompt):
                messages = prompt.to_chat_format()
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            completion = openai.ChatCompletion.create(
                model=self.ENGINE,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
                stop=stop,
                max_tokens=max_tokens,
            )

            # Convert ChatCompletion -> Completion format
            for choice in completion["choices"]:
                choice["text"] = choice["message"]["content"]
        else:
            completion = openai.Completion.create(
                model=self.ENGINE,
                prompt=str(prompt),
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
                stop=stop,
                max_tokens=max_tokens,
                logprobs=logprobs,
            )

        return completion

    def count_tokens_gpt2(self, text):
        # TODO(gg): Consider preprocessing to collapse whitespace, which could
        # bring the behavior more in line with the Codex tokenizer.
        return len(self.tokenizer(text, truncation=False)["input_ids"])
