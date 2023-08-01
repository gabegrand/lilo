"""
library_namer.py | Author : Catherine Wong and Gabe Grand.

Queries Codex to generate names for library functions.
"""
from typing import Dict, List, Optional

import numpy as np

import src.models.model_loaders as model_loaders
from dreamcoder.program import Invented
from dreamcoder.type import *
from src.experiment_iterator import SKIPPED_MODEL_FN
from src.models.gpt_base import *
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import ALL, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_NAMER]


class LibraryNamerPrompt(BasePrompt):

    TEXT_LIBRARY_HEADER = "You are writing software documentation. Your goal is to write human-readable names for the following library functions:"
    TEXT_ABSTRACTION_HEADER = "Consider the following anonymous function:"
    TEXT_EXAMPLES_HEADER = "Here are some examples of its usage:"

    def __init__(
        self,
        abstraction_definitions: Dict,
        abstraction_target: str,
        usage_examples: List[Dict],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
    ):
        self.abstraction_definitions = abstraction_definitions
        self.abstraction_target = abstraction_target
        self.usage_examples = usage_examples
        self.line_separator = line_separator
        self.message_list = []

    def __str__(self):
        self.to_message_list()
        return (
            self.DEFAULT_MESSAGE_SEPARATOR.join(
                [x["content"] for x in self.message_list]
            )
            + "\n"
        )

    def to_dict(self):
        return {
            "abstraction_definitions": self.abstraction_definitions,
            "abstraction_target": self.abstraction_target,
            "usage_examples": self.usage_examples,
        }

    @staticmethod
    def load_from_dict(d):
        return LibraryNamerPrompt(**d)

    def _fn_docstring(self, abstraction):
        definitions = self.abstraction_definitions[abstraction]
        docstring = f"{definitions['fn_name']} :: {definitions['fn_type']}\n"
        if definitions["fn_body"] is not None:
            docstring += f"\n{definitions['fn_body']}"
        if definitions["fn_description"] is not None:
            docstring += f"\ndescription: {definitions['fn_description']}"
        return docstring

    def _build_library_header(self):
        text_list = [self.TEXT_LIBRARY_HEADER + self.line_separator]
        for abstraction in self.abstraction_definitions.keys():
            text_list += [self._fn_docstring(abstraction) + self.line_separator]
        return self.line_separator.join(text_list)

    def _build_target_prompt(self):
        text_list = [self.TEXT_ABSTRACTION_HEADER + self.line_separator]
        text_list += [self._fn_docstring(self.abstraction_target) + self.line_separator]
        text_list += [self.TEXT_EXAMPLES_HEADER + self.line_separator]
        for example in self.usage_examples:
            text_list += [
                self.DEFAULT_PREFIX_LANGUAGE
                + example["language"]
                + self.line_separator
                + self.DEFAULT_PREFIX_PROGRAM
                + example["program"]
                + self.line_separator
            ]
        text_list += [
            self.make_abstraction_footer(
                self.abstraction_definitions[self.abstraction_target]["fn_name"]
            )
        ]
        return self.line_separator.join(text_list)

    def make_abstraction_footer(self, fn_name_numeric):
        return (
            f"Please write a human-readable name and description for `{fn_name_numeric}` in the JSON format shown below."
            + "\n"
            f"Your `readable_name` should be underscore-separated and should not contain any spaces."
            + "\n"
            f"It should also be unique (not existing in the function library above)."
            + "\n"
            f"If you cannot come up with a good name, please set `readable_name` to `null`."
            + "\n\n"
            "{" + "\n"
            f'    "anonymous_name": "{fn_name_numeric}",' + "\n"
            f'    "readable_name": TODO,' + "\n"
            f'    "description": TODO' + "\n"
            "}"
        )

    def to_message_list(self):
        message_list = [self.chat_message(self._build_library_header())]
        message_list += [self.chat_message(self._build_target_prompt())]
        self.message_list = message_list

    def to_chat_format(self):
        self.to_message_list()
        return self.message_list


@ModelRegistry.register
class GPTLibraryNamer(GPTBase, model_loaders.ModelLoader):
    name = "gpt_library_namer"

    results_file = "gpt_library_namer_results.json"

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
        body_task_selection,
        task_split: str,
        task_batch_ids: list,
        # Querying
        best_of: int = 1,
        n_samples_per_abstraction: int = 5,
        top_p: float = 0.1,
        max_tokens: int = 256,
        # Prompt construction
        n_usage_examples: int = 10,
        # Resume from prior runs
        resume_strategy: str = None,
        # Utilities
        verbose: bool = True,
    ):
        assert task_split == TRAIN
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        # Optional load from results JSON
        if self._maybe_load_from_checkpoint(
            experiment_state, task_split, resume_strategy
        ):
            return {
                SKIPPED_MODEL_FN: True,
            }

        abstraction_definitions = self._get_abstraction_definitions(experiment_state)
        abstraction_to_readable = {}

        for abstraction in abstraction_definitions.keys():

            # Update to have latest names
            abstraction_definitions = self._get_abstraction_definitions(
                experiment_state
            )
            usage_examples = self._get_usage_examples(
                experiment_state, abstraction, n_usage_examples
            )
            prompt = LibraryNamerPrompt(
                abstraction_definitions=abstraction_definitions,
                abstraction_target=abstraction,
                usage_examples=usage_examples,
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

            # Parse response
            parse_results = self._parse_completion(completion)
            selected_result = self._select_result(parse_results)

            if verbose:
                print(f"GPT ({self.ENGINE}) completion:")
                print(json.dumps(parse_results, indent=4))

            # Update function name
            if selected_result is not None:
                readable_name = selected_result["data"]["readable_name"]
                grammar.set_function_name(
                    str(abstraction),
                    name_class=LAPSGrammar.HUMAN_READABLE,
                    name=selected_result["data"]["readable_name"],
                )
                grammar.set_function_description(
                    name=str(abstraction),
                    description=selected_result["data"]["description"],
                )

                abstraction_to_readable[str(abstraction)] = selected_result["data"]
                abstraction_to_readable[str(abstraction)][
                    "usage_examples"
                ] = usage_examples

                print(
                    f"✅ Successfully named {abstraction_definitions[abstraction]['fn_name']} -> {readable_name}"
                )
                print(json.dumps(selected_result, indent=4))

            else:
                abstraction_to_readable[str(abstraction)] = None
                print(
                    f"❌ Failed to name {abstraction_definitions[abstraction]['fn_name']}"
                )

        n_abstractions_named = len(
            [x for x in abstraction_to_readable.values() if x is not None]
        )

        # TODO: Log/save outputs
        print("-" * 12)
        print(
            f"Completed library naming: {n_abstractions_named} / {len(abstraction_definitions)} abstractions successfully named."
        )
        print(json.dumps(abstraction_to_readable, indent=4))

        results = {
            "params": {
                "best_of": best_of,
                "n_samples_per_abstraction": n_samples_per_abstraction,
                "n_usage_examples": n_usage_examples,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            "summary": {
                "n_abstractions_named": n_abstractions_named,
                "n_abstractions_total": len(abstraction_definitions),
            },
            "abstractions": abstraction_to_readable,
        }

        # Save results to file
        results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.results_file,
        )
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
        with open(results_filepath, "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"Wrote results: {results_filepath}")

    def _maybe_load_from_checkpoint(
        self, experiment_state, task_split, resume_strategy, add_primitive=False
    ):
        if (resume_strategy == "first" and experiment_state.is_first_iteration()) or (
            resume_strategy == "every"
        ):
            # If RESUME_CHECKPOINT_DIRECTORY not defined, default to self checkpoint directory
            results_filepath_ext = os.path.join(
                os.getcwd(),
                experiment_state.get_checkpoint_directory_maybe_resume(),
                task_split,
                self.results_file,
            )
            if os.path.exists(results_filepath_ext):
                with open(results_filepath_ext, "r") as f:
                    results_json = json.load(f)

                # Update experiment state from file
                grammar = experiment_state.models[model_loaders.GRAMMAR]

                for abstraction, data in results_json["abstractions"].items():
                    if data is not None:
                        if add_primitive:
                            p = Invented.parse(str(abstraction))
                            new_productions = (0.0, p.infer(), p)
                            new_grammar = LAPSGrammar(
                                logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
                                productions=grammar.productions + [new_productions],
                                continuationType=grammar.continuationType,
                                initialize_parameters_from_grammar=grammar,
                            )
                            grammar = new_grammar
                        grammar.set_function_name(
                            str(abstraction),
                            name_class=LAPSGrammar.HUMAN_READABLE,
                            name=data["readable_name"],
                        )
                        grammar.set_function_description(
                            name=str(abstraction),
                            description=data["description"],
                        )

                # Copy external results file to checkpoint directory
                results_filepath = os.path.join(
                    os.getcwd(),
                    experiment_state.get_checkpoint_directory(),
                    task_split,
                    self.results_file,
                )
                os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
                with open(results_filepath, "w") as f:
                    json.dump(results_json, f, indent=4)

                print(f"{self.name}: Loaded results from {results_filepath_ext}")
                return True
            else:
                print(f"{self.name}: Results not found at {results_filepath_ext}")
                # if experiment_state.is_first_iteration():
                #     raise ValueError("Unable to resume first iteration.")
                return False

    def _get_numeric_name(self, experiment_state, abstraction):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        return grammar.get_name(
            str(abstraction), name_classes=[LAPSGrammar.NUMERIC_FUNCTION_NAMES]
        )

    def _get_abstraction_definitions(
        self, experiment_state, abstractions_only: bool = True
    ):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        if not abstractions_only:
            abstractions = [p for p in grammar.primitives]
        else:
            abstractions = [p for p in grammar.primitives if p.isInvented]

        abstraction_definitions = {}
        # for abstraction in sorted(abstractions, key=lambda p: str(p)):
        for abstraction in abstractions:
            abstraction_definitions[abstraction] = self._get_abstraction_definition(
                experiment_state, abstraction
            )

        return abstraction_definitions

    def _get_abstraction_definition(self, experiment_state, abstraction):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        fn_name = grammar.get_name(
            str(abstraction),
            name_classes=[
                LAPSGrammar.HUMAN_READABLE,
                LAPSGrammar.NUMERIC_FUNCTION_NAMES,
            ],
        )
        if abstraction.isInvented:
            fn_body = str(
                grammar.show_program(
                    str(abstraction)[
                        1:
                    ],  # Remove leading `#` so that any inlined abstractions are replaced with their fn_name
                    name_classes=[
                        LAPSGrammar.HUMAN_READABLE,
                        LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                    ],
                )
            )
            fn_description = grammar.get_function_description(abstraction)
        else:
            fn_body = None
            fn_description = None
        return {
            "fn_name": fn_name,
            "fn_body": fn_body,
            "fn_type": abstraction.infer(),
            "fn_description": fn_description,
        }

    # add documentation
    def _get_usage_examples(
        self, experiment_state, abstraction: Optional[Invented], n_usage_examples: int
    ):
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        tasks = list(experiment_state.task_frontiers[TRAIN].keys())
        rng.shuffle(tasks)

        usage_examples = []

        for task in tasks:
            frontier = experiment_state.task_frontiers[TRAIN][task]
            task_language = rng.choice(
                experiment_state.get_language_for_ids(TRAIN, [task.name])[0]
            )
            for e in frontier.entries:
                if (abstraction is None) or (str(abstraction) in e.tokens):
                    usage_examples += [
                        {
                            "task_name": task.name,
                            "program": grammar.show_program(
                                e.program,
                                name_classes=[
                                    LAPSGrammar.HUMAN_READABLE,
                                    LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                                ],
                                debug=True,
                            ),
                            "language": task_language,
                        }
                    ]
                if len(usage_examples) == n_usage_examples:
                    return usage_examples

                # Go to next task
                break

        return usage_examples

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
