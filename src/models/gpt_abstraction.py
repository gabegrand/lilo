"""
gpt_abstraction.py | Author : Maxine Liu.

Queries Codex to generate abstraction for functions.
"""
from typing import Dict, List

import src.models.model_loaders as model_loaders
from dreamcoder.type import *
from src.models.gpt_base import *

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_ABSTRACTION]


class LibraryAbstractionPrompt(BasePrompt):

    TEXT_ABSTRACTION_HEADER = "You are implementing abstraction learning. Your goal is to identify common patterns or sequences of operations in code and encapsulate them into reusable functions:"
    TEXT_FUNCTION_HEADER = "Consider the following functions in the library:"
    TEXT_EXAMPLES_HEADER = "Here are some examples of their usages:"

    def __init__(
        self,
        abstraction_definitions: Dict,
        usage_examples: List[Dict],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
    ):
        self.abstraction_definitions = abstraction_definitions
        self.usage_examples = usage_examples
        self.line_separator = line_separator

    def __str__(self):
        return (
            self.DEFAULT_MESSAGE_SEPARATOR.join(
                [x["content"] for x in self.to_message_list()]
            )
            + "\n"
        )

    # not sure if I need this
    def to_dict(self):
        return {
            "abstraction_definitions": self.abstraction_definitions,
            "usage_examples": self.usage_examples,
        }

    @staticmethod
    def load_from_dict(d):
        return LibraryAbstractionPrompt(**d)

    def _fn_docstring(self, abstraction):
        definitions = self.abstraction_definitions[abstraction]
        docstring = (
            f"{definitions['fn_name']} :: {definitions['fn_type']}\n"
            + f"{definitions['fn_body']}"
        )
        if definitions["fn_description"] is not None:
            docstring += f"\ndescription: {definitions['fn_description']}"
        return docstring

    def _build_abstraction_header(self):
        text_list = [
            self.TEXT_ABSTRACTION_HEADER
            + self.TEXT_FUNCTION_HEADER
            + self.line_separator
        ]
        for abstraction in self.abstraction_definitions.keys():
            text_list += [self._fn_docstring(abstraction) + self.line_separator]
        return self.line_separator.join(text_list)

    def _build_usage_prompt(self):
        text_list = [self.TEXT_EXAMPLES_HEADER + self.line_separator]
        for example in self.usage_examples:
            text_list += [
                self.DEFAULT_PREFIX_LANGUAGE
                + example["language"]
                + self.line_separator
                + self.DEFAULT_PREFIX_PROGRAM
                + example["program"]
                + self.line_separator
            ]

        # need to change here
        text_list += [self.make_abstraction_footer()]
        return self.line_separator.join(text_list)

    def make_abstraction_footer(self):
        return (
            f"Please write a compressed resuable function derived from the function library in the JSON format shown below."
            + "\n"
            f"It should be unique (not existing in the function library above)." + "\n"
            f"If you cannot come up with a good compressed function, return nothing."
            + "\n\n"
            "{" + "\n"
            # f'    "anonymous_name": "{fn_name_numeric}",' + "\n"
            # f'    "readable_name": TODO,' + "\n"
            # f'    "description": TODO' + "\n"
            "}"
        )

    def to_message_list(self):
        message_list = [self.chat_message(self._build_abstraction_header())]
        message_list += [self.chat_message(self._build_usage_prompt())]
        return message_list

    def to_chat_format(self):
        return self.to_message_list()
