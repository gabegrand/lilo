"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""
import os

import src.models.model_loaders as model_loaders
from src.models.stitch_base import StitchBase

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.PROGRAM_REWRITER]


@ModelRegistry.register
class StitchProgramRewriter(StitchBase, model_loaders.ModelLoader):
    name = "stitch_rewriter"

    # Programs for Stitch to rewrite
    stitch_program_file = "programs_to_rewrite.json"

    # Inventions from prior run of Stitch to use in rewriting process
    stitch_inventions_file = "stitch_output.json"

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
        program_file = os.path.join(os.getcwd(), self.stitch_program_file)
        inventions_file = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            self.stitch_inventions_file,
        )
        self.run_binary(
            bin="rewrite",
            stitch_kwargs={
                "program-file": program_file,
                "inventions-file": inventions_file,
            },
        )
