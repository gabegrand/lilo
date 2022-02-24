"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""
import src.models.model_loaders as model_loaders

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.PROGRAM_REWRITER]


@ModelRegistry.register
class StitchProgramRewriter(model_loaders.ModelLoader):
    name = "stitch_rewriter"

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
        pass
