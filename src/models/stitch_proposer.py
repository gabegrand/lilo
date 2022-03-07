"""
stitch_proposer.py | Author : Catherine Wong.

Library learning model that uses the Stitch compressor to propose libraries.
Expects an experiment_state with a GRAMMAR and FRONTIERs.
Updates GRAMMAR based on Stitch compression.
"""
import os

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

    stitch_input_frontiers_file = "stitch_frontiers.json"
    stitch_output_file = "stitch_output.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return StitchProposerLibraryLearner(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def get_compressed_grammar_mdl_prior_rank(
        self, experiment_state, task_splits, task_ids_in_splits, **kwargs
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) based on the training data description length to rerank the libraries.
        """
        # Write frontiers for stitch.
        input_frontiers_file = self.write_frontiers_to_file(
            experiment_state, task_splits, task_ids_in_splits
        )

        # Call stitch compressor.
        inv_programs = self._get_stitch_libraries(
            experiment_state,
            input_frontiers_file,
            max_arity=kwargs["max_arity"],
            iterations=kwargs["iterations"],
            candidates_per_iteration=kwargs["candidates_per_iteration"],
        )

        # Update the grammar with the new inventions.
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        new_productions = [(0.0, p.infer(), p) for p in inv_programs]
        new_grammar = LAPSGrammar(
            logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
            productions=grammar.productions + new_productions,
            continuationType=grammar.continuationType,
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
        input_frontiers_file,
        max_arity,
        iterations,
        candidates_per_iteration,
    ):
        input_file = os.path.join(os.getcwd(), input_frontiers_file)
        output_file = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.stitch_output_file,
        )
        self.run_binary(
            bin="compress",
            stitch_args=[input_file],
            stitch_kwargs={
                "out": output_file,
                "max-arity": max_arity,
                "iterations": iterations,
            },
        )
        inv_name_to_dc_fmt = self.get_inventions_from_file(
            stitch_output_file=output_file
        )
        inv_programs = [Program.parse(p) for p in inv_name_to_dc_fmt.values()]
        return inv_programs
