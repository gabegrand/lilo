"""
stitch_proposer.py | Author : Catherine Wong.

Library learning model that uses the Stitch compressor to propose libraries.
Expects an experiment_state with a GRAMMAR and FRONTIERs.
Updates GRAMMAR based on Stitch compression.
"""

import src.models.model_loaders as model_loaders
from dreamcoder.program import Program
from src.models.laps_grammar import LAPSGrammar
from src.models.stitch_base import StitchBase

LibraryLearnerRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LIBRARY_LEARNER
]


@LibraryLearnerRegistry.register
class StitchProposerLibraryLearner(StitchBase, model_loaders.ModelLoader):

    name = "stitch_proposer"

    frontiers_filename = "stitch_frontiers.json"
    inventions_filename = "stitch_output.json"

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
        update_grammar: bool = True,
        **kwargs
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) based on the training data description length to rerank the libraries.

        params:
            `update_grammar`: If True, updates the grammar in the experiment_state
                with the new inventions from compression. If False, runs compression
                and writes an inventions file, but leaves the grammar unaltered.
        """
        split = "_".join(task_splits)

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
        inventions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.inventions_filename,
            split=split,
        )
        self.run_binary(
            bin="compress",
            stitch_args=[frontiers_filepath, "--utility-by-rewrite", "--no-other-util"],
            stitch_kwargs={
                "out": inventions_filepath,
                "max-arity": max_arity,
                "iterations": iterations,
            },
        )
        inv_name_to_dc_fmt = self.get_inventions_from_file(
            stitch_output_filepath=inventions_filepath
        )
        inv_programs = [Program.parse(p) for p in inv_name_to_dc_fmt.values()]
        return inv_programs

    def get_inventions_and_metadata_for_current_iteration(self, experiment_state):
        inventions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.inventions_filename,
        )
        return self.get_inventions_and_metadata_from_file(inventions_filepath)
