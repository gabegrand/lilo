"""
library_namer.py | Author : Catherine Wong.

Queries Codex to generate names for library functions.
"""
import numpy as np
import src.models.model_loaders as model_loaders
from src.task_loaders import ALL
from src.models.codex_base import *

from dreamcoder.type import *

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_NAMER]

DEFAULT_HEADER = ""


@ModelRegistry.register
class CodexLibraryNamer(CodexBase, model_loaders.ModelLoader):
    name = "codex_library_namer"
    LIBRARY_DEFAULT_SEPARATOR = "\n"

    DEFAULT_COMMENT_HEADER = '"""Assign verbose, human-readable function names to functions based on their body definition and usage."""\n'
    DEFAULT_EXAMPLE_NAMED_INVENTIONS = "# List domain.\n#"
    DEFAULT_BASE_DSL_HEADER = '"""First, here are the original primitives in the programming language, commented with their functionality and verbose, human readable names."""\n'
    DEFAULT_INVENTION_HEADER = '"""Now, assign a verbose, human readable name to a new function defined using primitives in the programming language, based on the function body and how it is used in other programs."""\n'

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
        n_samples_per_invention: int = 5,
        n_invention_ids=ALL,
        use_comment_header=DEFAULT_COMMENT_HEADER,  # Header explaining the task.
        use_example_named_inventions: bool = False,  # Include examples of named inventions.
        use_base_dsl: bool = True,  # Include base DSL. Excludes constants.
        use_task_language: bool = False,  # Include names of programs
        n_train_programs_per_prompt: int = 10,  # How many programs to show where it was used.
        temperature: float = 0.75,
        max_tokens: int = 256,
        separator: str = LIBRARY_DEFAULT_SEPARATOR,
        engine: str = CodexBase.DEFAULT_ENGINE,
        debug: bool = False,
        use_pretty_naming: bool = True,
    ):
        """
        Queries Codex API to generate new names for library functions.
        
        """
        invention_metadatas = self._get_invention_metadata_for_n_inventions(
            experiment_state, n_invention_ids
        )
        fixed_prompt_header = self._build_fixed_prompt_header(
            experiment_state,
            use_comment_header,
            use_example_named_inventions,
            use_base_dsl,
        )
        for invention_metadata in invention_metadatas:
            invention_prompt = self._build_invention_prompt(
                experiment_state,
                invention_metadata,
                use_task_language,
                n_train_programs_per_prompt,
                use_pretty_naming=use_pretty_naming,
            )
            prompt = fixed_prompt_header + invention_prompt
            if debug:
                pass
            else:
                completion = self.query_codex(
                    prompt,
                    n_samples=n_samples_per_invention,
                    top_p=0.1,
                    temperature=None,
                    max_tokens=max_tokens,
                    engine=engine,
                    separator=separator,
                    logprobs=1,
                )
                if completion is not None:
                    # Sort by logprobs.
                    alternate_names = [
                        (choice["text"], np.mean(choice["logprobs"]["token_logprobs"]))
                        for choice in completion["choices"]
                    ]
                    alternate_names = sorted(
                        alternate_names, key=lambda c: c[-1], reverse=True
                    )
                    import pdb

                    pdb.set_trace()

        # Later on: we'll need a way to associate them *with* anything - use the stitch name as the ID.

    def _build_fixed_prompt_header(
        self,
        experiment_state,
        use_comment_header,
        use_example_named_inventions,
        use_base_dsl,
        skip_types=["int"],
    ):
        prompt_header = ""
        if use_comment_header is not None:
            prompt_header += use_comment_header
        if use_example_named_inventions:
            prompt_header += CodexLibraryNamer.DEFAULT_EXAMPLE_NAMED_INVENTIONS
        if use_base_dsl:
            prompt_header += CodexLibraryNamer.DEFAULT_BASE_DSL_HEADER
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            for p in grammar.primitives:
                if p.isInvented:
                    # TODO: add inventions here.
                    continue
                if str(p.tp) in skip_types:
                    continue
                prompt = "# Original function name: \n"
                prompt += f"{p}\n"
                prompt += f"# Functionality: {p.function_comment}\n"
                prompt += f"# Give an alternate verbose, human-readable name for this function that describes what it does. Prefix it with {grammar.function_prefix}_ \n"
                prompt += f"{p.alternate_names[-1]}"
                prompt += "\n\n"
                prompt_header += prompt

        return prompt_header

    def _build_invention_prompt(
        self,
        experiment_state,
        invention_metadata,
        use_task_language,
        n_train_programs_per_prompt,
        use_pretty_naming,
    ):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        prompt = ""
        prompt += CodexLibraryNamer.DEFAULT_INVENTION_HEADER
        prompt += "# Original function name: \n"
        prompt += invention_metadata["name"] + "\n"
        prompt += "# Function body: \n"
        if n_train_programs_per_prompt > 0:
            prompt += (
                f"# Here are {n_train_programs_per_prompt} examples of its usage: "
                + "\n"
            )
            prompt += (
                "\n".join(invention_metadata["rewritten"][:n_train_programs_per_prompt])
                + "\n"
            )
        prompt += invention_metadata["body"] + "\n"
        prompt += f"# Give an alternate verbose, human-readable name for this function that describes what it does. Prefix it with {grammar.function_prefix}_ \n"
        prompt += f"{grammar.function_prefix}_"

        if use_pretty_naming:
            prompt = prompt.replace("(lam", "(lambda")
            prompt = prompt.replace("inv", "fn_")
        return prompt

    def _get_invention_metadata_for_n_inventions(
        self, experiment_state, n_invention_ids
    ):
        library_learner = experiment_state.models[model_loaders.LIBRARY_LEARNER]
        try:
            inventions = library_learner.get_inventions_and_metadata_for_current_iteration(
                experiment_state
            )
            invention_keys = sorted(list(inventions.keys()))
            invention_keys = (
                invention_keys[:n_invention_ids]
                if n_invention_ids != ALL
                else invention_keys
            )
            return [inventions[k] for k in invention_keys]
        except:
            print("Could not load inventions and metadata from library learning model.")
            assert False

