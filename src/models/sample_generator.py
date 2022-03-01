"""
sample_generator.py | Author: Gabe Grand.

Queries Codex to generate new samples based on existing samples.

"""
import src.models.model_loaders as model_loaders

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.SAMPLE_GENERATOR]


@ModelRegistry.register
class CodexSampleGenerator(model_loaders.ModelLoader):
    name = "codex_sample_generator"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexSampleGenerator(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_samples(
        self, experiment_state, task_splits, task_ids_in_splits
    ):
        """
        Queries Codex API to generate new samples.
        """
        pass
