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

DEFAULT_HEADER = ""


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
                    separator=separator,
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
