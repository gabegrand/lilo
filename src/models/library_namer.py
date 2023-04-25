"""
library_namer.py | Author : Catherine Wong.

Queries Codex to generate names for library functions.
"""
import numpy as np

import src.models.model_loaders as model_loaders
from dreamcoder.type import *
from src.models.gpt_base import *
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import ALL, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_NAMER]


class LibraryNamerPrompt(BasePrompt):

    TEXT_DSL_HEADER = (
        "Our goal is to write human-readable names for the following anonymous functions:"
        + DEFAULT_LINE_SEPARATOR
    )
    TEXT_ABSTRACTION_HEADER = (
        "Consider the following anonymous function:" + DEFAULT_LINE_SEPARATOR
    )
    TEXT_EXAMPLES_HEADER = (
        "Here are some examples of its usage:" + DEFAULT_LINE_SEPARATOR
    )

    def __init__(
        self,
        text_header,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
    ):
        self.text_header = text_header
        self.chat_history = []

        self.line_separator = line_separator

    def __str__(self):
        return (
            self.DEFAULT_MESSAGE_SEPARATOR.join(
                [x["content"] for x in self.to_message_list()]
            )
            + "\n"
        )

    def to_dict(self):
        pass

    def load_from_dict(self):
        pass

    def make_abstraction_footer(self, fn_name_numeric):
        return (
            f"Please write a human-readable name and description for `{fn_name_numeric}` in the JSON format shown below."
            + "\n"
            f"Your `readable_name` should be underscore-separated and should not contain any spaces."
            + "\n"
            f"If you cannot come up with a good name, please set `readable_name` to `null`."
            + "\n\n"
            "{" + "\n"
            f'    "anonymous_name": "{fn_name_numeric}",' + "\n"
            f'    "readable_name": TODO,' + "\n"
            f'    "description": TODO' + "\n"
            "}"
        )

    def add_abstraction_prompt(
        self, text_abstraction_definition, text_abstraction_examples, fn_name_numeric
    ):
        message = self.chat_message(
            self.TEXT_ABSTRACTION_HEADER
            + self.line_separator
            + text_abstraction_definition
            + self.line_separator
            + self.TEXT_EXAMPLES_HEADER
            + self.line_separator
            + text_abstraction_examples
            + self.line_separator
            + self.make_abstraction_footer(fn_name_numeric)
        )
        self.chat_history += [message]

    def add_abstraction_reply(self, fn_naming_data):
        message = self.chat_message(
            text=json.dumps(fn_naming_data, indent=4),
            role="assistant",
        )
        self.chat_history += [message]

    def to_message_list(self):
        message_list = [self.chat_message(self.TEXT_DSL_HEADER + self.text_header)]
        message_list += self.chat_history
        return message_list

    def to_chat_format(self):
        return self.to_message_list()


@ModelRegistry.register
class GPTLibraryNamer(GPTBase, model_loaders.ModelLoader):
    name = "gpt_library_namer"

    ERROR_JSON = "error_json"
    ERROR_MISSING_FIELD = "error_missing_field"
    ERROR_NULL_NAME = "error_null_name"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTLibraryNamer(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(engine=engine)

    def generate_library_names(
        self,
        experiment_state,
        task_split: str,
        task_batch_ids: list,
        # Querying
        best_of: int = 1,
        n_samples_per_abstraction: int = 5,
        top_p: float = 0.1,
        max_tokens: int = 256,
        # Prompt construction
        n_usage_examples: int = 5,
        # Utilities
        verbose: bool = True,
    ):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        abstractions_to_name = self._get_abstractions_to_name(experiment_state)

        # TODO: Optional load from cache

        # Build prompt header
        prompt_text_header = self._build_prompt_header(
            experiment_state, abstractions_to_name
        )
        prompt = LibraryNamerPrompt(prompt_text_header)

        for abstraction in abstractions_to_name:

            # Build abstraction prompt
            fn_name_numeric = self._get_numeric_name(experiment_state, abstraction)
            (
                text_abstraction_definition,
                text_abstraction_examples,
            ) = self._build_prompt_abstraction(
                experiment_state, abstraction, n_usage_examples=n_usage_examples
            )
            prompt.add_abstraction_prompt(
                text_abstraction_definition, text_abstraction_examples, fn_name_numeric
            )

            if verbose:
                print(prompt)

            # Query
            completion = self.query_completion(
                prompt,
                n_samples=n_samples_per_abstraction,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=None,
            )
            if verbose:
                print(completion)

            # Parse response
            parse_results = self._parse_completion(completion)
            selected_result = self._select_result(parse_results)

            # Update function name
            if selected_result is not None:
                grammar.set_function_name(
                    str(abstraction),
                    name_class=LAPSGrammar.HUMAN_READABLE,
                    name=selected_result["data"]["readable_name"],
                )

                # Update prompt
                prompt.add_abstraction_reply(selected_result["data"])

                print(
                    f"✅ Successfully named {fn_name_numeric} -> {selected_result['data']['readable_name']}"
                )
                print(selected_result)

            else:
                print(f"❌ Failed to name {fn_name_numeric}")

        # TODO: Log/save outputs

    def _parse_completion(self, completion):
        parse_results = []
        for choice in completion["choices"]:
            try:
                data = json.loads(choice["text"])
            except:
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryNamer.ERROR_JSON,
                    }
                )
                continue

            if not (
                ("anonymous_name" in data)
                and ("readable_name" in data)
                and ("description" in data)
            ):
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryNamer.ERROR_MISSING_FIELD,
                    }
                )
                continue

            if not data["readable_name"]:
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryNamer.ERROR_NULL_NAME,
                    }
                )
                continue

            parse_results.append(
                {
                    "index": choice["index"],
                    "text": choice["text"],
                    "valid": True,
                    "data": data,
                }
            )
        return parse_results

    def _select_result(self, parse_results):
        for result in parse_results:
            # For now, just return the first valid result
            if result["valid"]:
                return result

    def _get_numeric_name(self, experiment_state, abstraction):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        return grammar.get_name(
            str(abstraction), name_classes=[LAPSGrammar.NUMERIC_FUNCTION_NAMES]
        )

    def _get_abstraction_definition(self, experiment_state, abstraction):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        fn_name_numeric = self._get_numeric_name(experiment_state, abstraction)
        fn_body = str(
            grammar.show_program(
                abstraction.betaNormalForm(),
                name_classes=[
                    LAPSGrammar.HUMAN_READABLE,
                    LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                ],
            )
        )
        return f"{fn_name_numeric} :: {abstraction.infer()}\n{fn_body}"

    def _build_prompt_header(self, experiment_state, abstractions_to_name):
        prompt_list = []
        for abstraction in abstractions_to_name:
            prompt_list += [
                self._get_abstraction_definition(experiment_state, abstraction)
            ]

        return (DEFAULT_LINE_SEPARATOR * 2).join(prompt_list)

    def _build_prompt_abstraction(
        self, experiment_state, abstraction, n_usage_examples
    ):
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        text_abstraction_definition = (
            self._get_abstraction_definition(experiment_state, abstraction)
            + DEFAULT_LINE_SEPARATOR
        )

        prompt_list = []
        example_usages = self._get_example_usages(
            experiment_state, abstraction, n_usage_examples
        )
        for task, program in example_usages.items():
            task_language = rng.choice(
                experiment_state.get_language_for_ids(TRAIN, [task.name])[0]
            )
            task_program = grammar.show_program(
                program,
                name_classes=[
                    LAPSGrammar.HUMAN_READABLE,
                    LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                ],
                debug=True,
            )
            prompt_list += [BasePrompt.DEFAULT_PREFIX_LANGUAGE + str(task_language)]
            prompt_list += [
                BasePrompt.DEFAULT_PREFIX_PROGRAM
                + str(task_program)
                + DEFAULT_LINE_SEPARATOR
            ]

        text_abstraction_examples = DEFAULT_LINE_SEPARATOR.join(prompt_list)

        return text_abstraction_definition, text_abstraction_examples

    def _get_example_usages(self, experiment_state, abstraction, n_usage_examples):
        """
        :ret: [(task, example) for n_usage_examples using the abstraction]
        """
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        example_usages = dict()
        tasks = list(experiment_state.task_frontiers[TRAIN].keys())
        rng.shuffle(tasks)
        for task in tasks:
            frontier = experiment_state.task_frontiers[TRAIN][task]
            for e in frontier.entries:
                if str(abstraction) in e.tokens and not task in example_usages:
                    example_usages[task] = e.program
                    if len(example_usages) == n_usage_examples:
                        return example_usages
        return example_usages

    def _get_abstractions_to_name(self, experiment_state):
        """
        :ret: [array of abstraction expressions to name]
        """
        # Get abstractions.
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        abstractions = [p for p in grammar.primitives if p.isInvented]
        abstractions = sorted(abstractions, key=lambda p: str(p))
        return abstractions


DEFAULT_HEADER = ""

# Deprecated: Use GPTLibraryNamer
@ModelRegistry.register
class CodexLibraryNamer(GPTBase, model_loaders.ModelLoader):
    name = "codex_library_namer"
    LIBRARY_DEFAULT_SEPARATOR = "\n"

    DEFAULT_COMMENT_HEADER = '"""Assign verbose, human-readable function names to functions based on their body definition and usage."""\n'
    DEFAULT_BASE_DSL_HEADER = '"""First, here are the original primitives in the programming language, commented with their functionality and verbose, human readable names."""\n'
    DEFAULT_INVENTION_HEADER = '"""Now, assign a verbose, human readable name to a new function defined using primitives in the programming language, based on the function body and how it is used in other programs."""\n'

    # Which library functions to rename.
    ALL_UNNAMED = "all_unnamed"

    # How to select names.
    TOP_1 = "top_1"
    SAMPLE_LOG_PROBS = "sample_log_probs"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexLibraryNamer(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_library_names(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: list,
        inventions_to_name=ALL_UNNAMED,
        n_codex_samples_per_invention: int = 5,
        name_selection_criteria: str = TOP_1,
        prompt_comment_header: str = DEFAULT_COMMENT_HEADER,
        prompt_with_base_dsl: bool = True,
        prompt_with_task_language: bool = False,
        prompt_with_n_example_programs: int = 10,
        input_name_classes: list = [LAPSGrammar.NUMERIC_FUNCTION_NAMES],
        output_name_class=LAPSGrammar.HUMAN_READABLE,  # What class of function names to append the new names to.
        body_name_class=LAPSGrammar.HUMAN_READABLE,  # What class of function names to use to show the function body.
        temperature: float = 0.75,
        max_tokens: int = 256,
        separator: str = LIBRARY_DEFAULT_SEPARATOR,
        debug: bool = False,
        verbose: bool = True,
    ):
        """
        Queries Codex API to generate new names for library functions.

        params:
            inventions_to_name: which inventions to generate names for: {ALL, ALL_UNNAMED}; ALL_UNNAMED only selects inventions w/out names in the output_name_class.
            n_codex_samples_per_invention: how many names to sample per invention; we choose 1 of these.
            name_selection_criteria: how to choose a name amongst samples. TOP_1: ranks by logprobs and choose 1; SAMPLE_LOG_PROBS samples as a categorical according to log probs.

            prompt_comment_header: String prompt that will prefix the Codex prompt for each invention.
            prompt_with_base_dsl: If true, includes example usage from the Base DSL.
            prompt_with_task_language: If true, includes example task annotations where available.
            prompt_with_n_example_programs: how many example usage programs to show for each primitive or invention.

            input_name_classes: List of name classes to show the initial functions that need naming.
            output_name_class: Desired name class for the generated names.
            body_name_class: List of name classes to show the body of inventions.

            temperature, max_tokens, separator: see codex_base.
        """
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        # Gets inventions we want to rename and tasks where they are used.
        inventions = self._get_inventions_to_name(
            experiment_state, inventions_to_name, output_name_class
        )
        # Builds the prompt header for each invention including the Base DSL
        # TODO (catwong): Update this prompt to match the style of the invention prompt.
        fixed_prompt_header = self._build_fixed_prompt_header(
            experiment_state,
            prompt_comment_header,
            prompt_with_base_dsl,
        )
        for invention in inventions:
            # TODO (catwong): Implement use_task_language to include task annotations.
            invention_prompt = self._build_invention_prompt(
                experiment_state,
                invention,
                prompt_with_task_language,
                prompt_with_n_example_programs,
                body_name_class=body_name_class,
                input_name_classes=input_name_classes,
                output_name_class=output_name_class,
            )
            prompt = fixed_prompt_header + invention_prompt
            print(prompt)
            if debug:
                # TODO (catwong): open space for debugging.
                pass
            else:
                completion = self.query_completion(
                    prompt,
                    n_samples=n_codex_samples_per_invention,
                    top_p=0.1,
                    temperature=None,
                    max_tokens=max_tokens,
                    logprobs=1,
                )
                if completion is not None:
                    # Sort by logprobs.
                    alternate_names = [
                        (choice["text"], np.mean(choice["logprobs"]["token_logprobs"]))
                        for choice in completion["choices"]
                    ]
                    alternate_name, log_prob = self._select_name(
                        alternate_names, name_selection_criteria
                    )
                    alternate_name = f"{grammar.function_prefix}_{alternate_name}"
                    alternate_name = grammar.set_function_name(
                        str(invention),
                        name_class=output_name_class,
                        name=alternate_name,
                    )
                    if verbose:
                        print(
                            f"Setting function name for {str(invention)} to {alternate_name} w/ log_p = {log_prob}"
                        )
        if verbose:
            # Display current grammar.
            print("Grammar after naming is now:")
            for p in grammar.primitives:
                print(
                    f"{grammar.function_names[str(p)][LAPSGrammar.NUMERIC_FUNCTION_NAMES]}: {grammar.function_names[str(p)][LAPSGrammar.DEFAULT_FUNCTION_NAMES]}"
                )
                print(
                    f"\t Human readable: {grammar.function_names[str(p)].get(LAPSGrammar.HUMAN_READABLE, 'UNNAMED')}"
                )
                print("\n")

    def _select_name(self, alternate_names, name_selection_criteria):
        alternate_names = sorted(alternate_names, key=lambda c: c[-1], reverse=True)
        if name_selection_criteria == self.TOP_1:
            return alternate_names[0]
        elif name_selection_criteria == self.SAMPLE_LOG_PROBS:
            # Sample according to probability.
            names, probabilities = zip(*alternate_names)
            return np.random.choice(alternate_names, p=probabilities)[0]
        else:
            assert False

    def _build_fixed_prompt_header(
        self,
        experiment_state,
        prompt_comment_header,
        prompt_with_base_dsl,
        skip_types=["int"],
    ):
        prompt_header = ""
        if prompt_comment_header is not None:
            prompt_header += prompt_comment_header
        if prompt_with_base_dsl:
            prompt_header += CodexLibraryNamer.DEFAULT_BASE_DSL_HEADER
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            for p in grammar.primitives:
                if p.isInvented:
                    # TODO: add previously named inventions here.
                    continue
                if str(p.tp) in skip_types:
                    continue
                prompt = "# Original function name: \n"  # TODO: use the numeric input name here.
                prompt += f"{p}\n"  # TODO: use the human readable names; give examples of usage - match the form.
                prompt += f"# Functionality: {p.function_comment}\n"
                prompt += f"# Give an alternate verbose, human-readable name for this function that describes what it does. Prefix it with {grammar.function_prefix}_ \n"
                prompt += (
                    f"{p.alternate_names[-1]}"  # TODO: use the human readable name.
                )
                prompt += "\n\n"
                prompt_header += prompt

        return prompt_header

    def _build_invention_prompt(
        self,
        experiment_state,
        invention,
        prompt_with_task_language,
        prompt_with_n_example_programs,
        body_name_class,
        input_name_classes,
        output_name_class,
    ):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        prompt = ""
        prompt += CodexLibraryNamer.DEFAULT_INVENTION_HEADER

        input_function_name = grammar.get_name(str(invention), input_name_classes)
        prompt += "# Original function name: \n"
        prompt += input_function_name + "\n"

        if prompt_with_n_example_programs > 0:
            example_usages = self._get_example_usages(
                experiment_state, invention, prompt_with_n_example_programs
            )
            prompt += (
                f"# Here are {prompt_with_n_example_programs} examples of its usage: "
                + "\n"
            )
            # TODO: add language; more intelligent example usage selection.
            example_programs = [
                grammar.show_program(
                    example,
                    name_classes=[body_name_class] + input_name_classes,
                    debug=True,
                )
                for example in example_usages.values()
            ]
            prompt += "\n".join(example_programs) + "\n"
        prompt += "# Function body: \n"
        function_body = str(
            grammar.show_program(
                invention.betaNormalForm(), name_classes=[body_name_class]
            )
        )
        prompt += function_body + "\n"
        prompt += f"# Give an alternate verbose, human-readable name for this function that describes what it does. Prefix it with {grammar.function_prefix}_ \n"
        prompt += f"{grammar.function_prefix}_"

        return prompt

    def _get_example_usages(self, experiment_state, primitive, n_examples):
        """
        :ret: [(task, example) for n_examples using the primitive]
        """
        # TODO: find examples where its not used along with inventions.
        example_usages = dict()
        for task, frontier in experiment_state.task_frontiers[TRAIN].items():
            for e in frontier.entries:
                if str(primitive) in e.tokens and not task in example_usages:
                    example_usages[task] = e.program
                    if len(example_usages) == n_examples:
                        return example_usages
        return example_usages

    def _get_inventions_to_name(
        self, experiment_state, inventions_to_name, output_name_class
    ):
        """
        :ret: [array of Invention expressions to name]
        """
        # Get inventions.
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        inventions = [p for p in grammar.primitives if p.isInvented]
        if inventions_to_name == ALL:
            pass
        elif inventions_to_name == self.ALL_UNNAMED:
            inventions = [
                i
                for i in inventions
                if not grammar.has_alternate_name(i, output_name_class)
            ]
        inventions = sorted(inventions, key=lambda p: str(p))
        return inventions
