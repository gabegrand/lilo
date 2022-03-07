"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""

import src.models.model_loaders as model_loaders
from src.models.stitch_base import StitchBase

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.PROGRAM_REWRITER]


@ModelRegistry.register
class StitchProgramRewriter(StitchBase, model_loaders.ModelLoader):
    name = "stitch_rewriter"

    # Programs for Stitch to rewrite
    programs_filename = "programs_to_rewrite.json"

    # Inventions from prior run of Stitch to use in rewriting process
    inventions_filename = "stitch_output.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return StitchProgramRewriter(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def get_rewritten_frontiers_for_grammar(
        self, experiment_state, task_splits, task_ids_in_splits
    ):
        """
        Updates experiment_state frontiers wrt. the experiment_state.models[GRAMMAR]
        """
        inventions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.inventions_filename,
        )
        programs_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.programs_filename,
        )
        self.write_frontiers_to_file(
            experiment_state,
            task_splits,
            task_ids_in_splits,
            frontiers_filepath=programs_filepath,
        )
        self.run_binary(
            bin="rewrite",
            stitch_kwargs={
                "program-file": programs_filepath,
                "inventions-file": inventions_filepath,
            },
        )
