"""
stitch_proposer.py | Author : Catherine Wong.

Library learning model that uses the Stitch compressor to propose libraries.
Expects an experiment_state with a GRAMMAR and FRONTIERs.
Updates GRAMMAR based on Stitch compression.
"""

import json

import stitch_core as stitch

import src.models.model_loaders as model_loaders
from dreamcoder.program import Invented
from src.models.laps_grammar import LAPSGrammar
from src.models.stitch_base import StitchBase

LibraryLearnerRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LIBRARY_LEARNER
]


@LibraryLearnerRegistry.register
class StitchProposerLibraryLearner(StitchBase, model_loaders.ModelLoader):

    name = "stitch_proposer"

    frontiers_filename = "stitch_frontiers.json"
    abstractions_filename = "stitch_output.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return StitchProposerLibraryLearner(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def get_compressed_grammar_mdl_prior_rank(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        include_samples,
        use_mdl_program: bool = True,
        beta_reduce_programs: bool = True,
        update_grammar: bool = True,
        replace_existing_abstractions: bool = True,
        **kwargs,
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) based on the training data description length to rerank the libraries.

        params:
            `use_mdl_program`: If True, compresses the single MDL program for each frontier.
                If False, compresses all programs in the frontier.
            `beta_reduce_programs`: Whether to beta reduce programs before compression.
                This will rewrite the programs into the base DSL, removing any abstractions.
            `update_grammar`: If True, updates the grammar in the experiment_state
                with the new inventions from compression. If False, runs compression
                and writes an inventions file, but leaves the grammar unaltered.
        """
        split = "_".join(task_splits)

        # Update the grammar to remove all existing abstractions.
        if update_grammar and replace_existing_abstractions:
            experiment_state.models[model_loaders.GRAMMAR] = LAPSGrammar.fromGrammar(
                experiment_state.models[model_loaders.GRAMMAR], remove_inventions=True
            )

        # Write frontiers for stitch.
        frontiers_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.frontiers_filename,
            split=split,
        )
        self.write_frontiers_to_file(
            experiment_state,
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            frontiers_filepath=frontiers_filepath,
            use_mdl_program=use_mdl_program,
            beta_reduce_programs=beta_reduce_programs,
            include_samples=include_samples,
        )

        # Call stitch compressor.
        inv_programs = self._get_stitch_libraries(
            experiment_state,
            frontiers_filepath,
            split,
            max_arity=kwargs["max_arity"],
            iterations=kwargs["iterations"],
            candidates_per_iteration=kwargs["candidates_per_iteration"],
        )

        # Update the grammar with the new inventions.
        if update_grammar:
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            new_productions = [(0.0, p.infer(), p) for p in inv_programs]
            new_grammar = LAPSGrammar(
                logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
                productions=grammar.productions + new_productions,
                continuationType=grammar.continuationType,
                initialize_parameters_from_grammar=grammar,
            )

            experiment_state.models[model_loaders.GRAMMAR] = new_grammar

            print(
                f"Updated grammar (productions={len(grammar.productions)}) with {len(new_productions)} new abstractions."
            )

    def get_compressed_grammar_lm_prior_rank(
        self, experiment_state, task_splits, task_ids_in_splits, max_arity, iterations
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) under a language model (default Codex) to rerank the libraries.
        """
        # grammar:
        # experiment_state.task_frontiers

    def get_compressed_grammar_lm_alignment_rank(
        self, experiment_state, task_splits, task_ids_in_splits, max_arity, iterations
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(language, libraries) under a language model (default Codex) to rerank the libraries.
        """
        # catwong: here's how you could get an example grammar.
        # catwong: here's how you get all of the language out of the

    def _get_stitch_libraries(
        self,
        experiment_state,
        frontiers_filepath,
        split,
        max_arity,
        iterations,
        candidates_per_iteration,
    ):
        abstractions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.abstractions_filename,
            split=split,
        )

        with open(frontiers_filepath, "r") as f:
            frontiers_dict = json.load(f)
            stitch_kwargs = stitch.from_dreamcoder(frontiers_dict)

        compression_result = stitch.compress(
            **stitch_kwargs,
            iterations=iterations,
            max_arity=max_arity,
            no_other_util=True,
        )
        abstractions = [
            Invented.parse(abs["dreamcoder"])
            for abs in compression_result.json["abstractions"]
        ]

        with open(abstractions_filepath, "w") as f:
            json.dump(compression_result.json, f)

        return abstractions
