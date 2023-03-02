"""
clevr: encoder.py | Author : Sam Acquaviva.

Same format as encoder.py in compositional_graphics, but for the Clevr domain.
"""

from src.experiment_iterator import *
from src.models.model_loaders import (
    ModelLoaderRegistries,
    EXAMPLES_ENCODER,
    ModelLoader,
)
from src.task_loaders import TRAIN, TEST, ALL

from dreamcoder.domains.clevr.clevrRecognition import ClevrFeatureExtractor

ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]

@ExamplesEncoderRegistry.register
class ClevrFeatureExamplesEncoder(ModelLoader):
    """Loads the Clevr Feature Extractor class. Note that this does not return an initialized model. 
    It returns the class that can be instantiated from the experiment state, with other params set. 
    Original source: dreamcoder/domains/clevr/clevrRecognition.py"""

    name = "clevr"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)
            return ClevrFeatureExtractor(
                tasks=all_train_tasks,
                testingTasks=all_test_tasks,
                **kwargs
            )

        return experiment_state_initializer
