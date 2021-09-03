"""
laps_dreamcoder_recognition.py | Author : Catherine Wong.

Utility wrapper function around the DreamCoder recognition model. Elevates common functions to be class functions and allows them to be called with an ExperimentState.
"""
from src.task_loaders import *
import src.models.model_loaders as model_loaders

from dreamcoder.recognition import RecognitionModel

AmortizedSynthesisModelRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.AMORTIZED_SYNTHESIS
]

# TODO: implement a model loader so that we can load it.

ACTIVATION_TANH, ACTIVATION_RELU = "tanh", "relu"


@AmortizedSynthesisModelRegistry.register
class LAPSDreamCoderRecognitionLoader(model_loaders.ModelLoader):
    name = "dreamcoder_recognition"

    def load_model(self):
        return LAPSDreamCoderRecognition()


class LAPSDreamCoderRecognition:
    """LAPSDreamCoderRecognition: containiner wrapper for a DreamCoder recognition model. The neural weights are fully reset and retrained when optimize_model_for_frontiers is called."""

    # Contain the neural recognition model. This is re-trained each time optimize_model_for_frontiers is called.
    def __init__(self):
        self._neural_recognition_model = None

    def infer_programs_for_tasks(self):
        """Searches from a new recognition model. Requires a trained recognition model from optimize_model_for_frontiers. Updates the experiment_state.models[AMORTIZED_SYNTHESIS] to contain the trained model."""
        pass

    def maybe_initialize_example_encoder(self, task_encoder_types, experiment_state):
        if model_loaders.EXAMPLES_ENCODER not in task_encoder_types:
            return None
        # Initialize from tasks.
        model_initializer_fn = experiment_state.models[model_loaders.EXAMPLES_ENCODER]
        return model_initializer_fn(experiment_state)

    def optimize_model_for_frontiers(
        self,
        experiment_state,
        task_split=TRAIN,
        task_batch_ids=ALL,
        task_encoder_types=[
            model_loaders.EXAMPLES_ENCODER
        ],  # Task encoders to use: [EXAMPLES_ENCODER, LANGUAGE_ENCODER]
        recognition_train_steps=5000,  # Gradient steps to train model.
        recognition_train_timeout=None,  # Alternatively, how long to train the model
        recognition_train_epochs=None,  # Alternatively, how many epochs to train
        helmholtz_ratio=0.5,  # How often to sample Helmholtz samples
        sample_evaluation_timeout=1.0,  # How long to spend trying to evaluate samples.
        matrix_rank=None,  # Maximum rank of bigram transition matrix for contextual recognition model. Defaults to full rank.
        mask=False,  # Unconditional bigram masking
        activation=ACTIVATION_TANH,
        contextual=True,
        bias_optimal=True,
        auxiliary_loss=True,
        cuda=False,
        cpus=12,
        max_mem_per_enumeration_thread=1000000,
        require_ground_truth_frontiers=False,
    ):
        """Trains a new recognition model with respect to the frontiers. Updates the experiment_state.models[AMORTIZED_SYNTHESIS] to contain the trained model."""

        # Skip training if no non-empty frontiers.
        if (
            require_ground_truth_frontiers
            and len(experiment_state.get_non_empty_frontiers_for_split(task_split)) < 1
        ):
            print(
                f"require_ground_truth_frontiers=True and no non-empty frontiers in {task_split}. skipping optimize_model_for_frontiers"
            )
            return
        # Initialize I/O example encoders.
        example_encoder = self.maybe_initialize_example_encoder(
            task_encoder_types, experiment_state
        )
        # Initialize the neural recognition model.
        self._neural_recognition_model = RecognitionModel(
            example_encoder=example_encoder,
            language_encoder=None,
            grammar=experiment_state.models[model_loaders.GRAMMAR],
            mask=mask,
            rank=matrix_rank,
            activation=activation,
            cuda=cuda,
            contextual=contextual,
            pretrained_model=None,
            helmholtz_nearest_language=0,
            helmholtz_translations=None,  # This object contains information for using the joint generative model over programs and language.
            nearest_encoder=None,
            nearest_tasks=[],
            id=0,
        )

        # Train the model.
        all_train_frontiers = experiment_state.get_frontiers_for_ids(
            task_split=task_split, task_ids=task_batch_ids
        )

        # Returns any existing samples in the experiment state
        def get_sample_frontiers():
            return experiment_state.get_frontiers_for_ids(
                task_split=task_split,
                task_ids=ALL,
                include_samples=True,
                include_ground_truth_tasks=False,
            )

        self._neural_recognition_model.train(
            all_train_frontiers,
            biasOptimal=bias_optimal,
            helmholtzFrontiers=get_sample_frontiers(),
            CPUs=cpus,
            evaluationTimeout=sample_evaluation_timeout,
            timeout=recognition_train_timeout,
            steps=recognition_train_steps,
            helmholtzRatio=helmholtz_ratio,
            auxLoss=auxiliary_loss,
            vectorized=True,
            epochs=recognition_train_epochs,
        )
