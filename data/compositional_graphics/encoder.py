"""
compositional_graphics: encoder.py | Author : Catherine Wong.

Utility function for loading the examples encoders for the compositional graphics domain. This encoder was originally designed and used in the LAPS-ICML 2021 paper and the Ellis et. al DreamCoder papers and can be found in the dreamcoder/logo domain.
"""
from src.experiment_iterator import *
from src.models.model_loaders import (
    ModelLoaderRegistries,
    EXAMPLES_ENCODER,
    ModelLoader,
)
from src.task_loaders import TRAIN, TEST, ALL

from dreamcoder.domains.logo.main import LogoFeatureCNN


ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]

# Temporary directory to store dream images.
LOGO_DREAMS_TMP_DIRECTORY = "dreams/"


@ExamplesEncoderRegistry.register
class LogoFeatureCNNExamplesEncoder(ModelLoader):
    """Loads the LOGO Feature CNN class. Note that this does not return an initialized model. It returns the class that can be instantiated from the experiment state, with other params set. Original source: dreamcoder/domains/logo/main.py"""

    name = "LOGO"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)

            # Make the local directory.
            local_dreams_directory = os.path.join(
                exp_state.metadata[EXPORT_DIRECTORY], LOGO_DREAMS_TMP_DIRECTORY
            )

            return LogoFeatureCNN(
                tasks=all_train_tasks,
                testingTasks=all_test_tasks,
                local_prefix_dreams=local_dreams_directory,
                **kwargs
            )

        return experiment_state_initializer
