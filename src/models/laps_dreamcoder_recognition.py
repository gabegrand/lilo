"""
laps_dreamcoder_recognition.py | Author : Catherine Wong.

Utility wrapper function around the DreamCoder recognition model. Elevates common functions to be class functions and allows them to be called with an ExperimentState.
"""
import src.models.model_loaders as model_loaders
from dreamcoder.recognition import RecognitionModel
from src.experiment_iterator import INIT_FRONTIERS_FROM_CHECKPOINT, SKIPPED_MODEL_FN
from src.task_loaders import *

AmortizedSynthesisModelRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.AMORTIZED_SYNTHESIS
]

# TODO: implement a model loader so that we can load it.

ACTIVATION_TANH, ACTIVATION_RELU = "tanh", "relu"


@AmortizedSynthesisModelRegistry.register
class LAPSDreamCoderRecognitionLoader(model_loaders.ModelLoader):
    name = "dreamcoder_recognition"

    def load_model(self, experiment_state):
        return LAPSDreamCoderRecognition()


class LAPSDreamCoderRecognition:
    """LAPSDreamCoderRecognition: containiner wrapper for a DreamCoder recognition model. The neural weights are fully reset and retrained when optimize_model_for_frontiers is called."""

    DEFAULT_MAXIMUM_FRONTIER = 5  # Maximum top-programs to keep in frontier
    DEFAULT_CPUS = os.cpu_count()  # Parallel CPUs
    DEFAULT_ENUMERATION_SOLVER = "ocaml"  # OCaml, PyPy, or Python enumeration
    DEFAULT_SAMPLER = "helmholtz"
    DEFAULT_BINARY_DIRECTORY = "dreamcoder"
    DEFAULT_EVALUATION_TIMEOUT = 1  # Timeout for evaluating a program on a task
    DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD = 1000000000  # Max memory usage per thread

    # Contain the neural recognition model. This is re-trained each time optimize_model_for_frontiers is called.
    def __init__(self):
        self._neural_recognition_model = None

    def infer_programs_for_tasks(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        enumeration_timeout,
        maximum_frontier=DEFAULT_MAXIMUM_FRONTIER,
        cpus=DEFAULT_CPUS,
        solver=DEFAULT_ENUMERATION_SOLVER,
        evaluation_timeout=DEFAULT_EVALUATION_TIMEOUT,
        max_mem_per_enumeration_thread=DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD,
        solver_directory=DEFAULT_BINARY_DIRECTORY,
        allow_resume=True,
    ):
        """
        Infers programs for tasks via top-down enumerative search from the grammar.
        Updates Frontiers in experiment_state with discovered programs.

        Wrapper function around recognition.enumerateFrontiers from dreamcoder.recognition.
        """
        if allow_resume and experiment_state.metadata[INIT_FRONTIERS_FROM_CHECKPOINT]:
            if experiment_state.maybe_resume_from_checkpoint():
                print(
                    f"infer_programs_for_tasks: Restored frontiers from checkpoint and skipped enumeration."
                )
                return {
                    SKIPPED_MODEL_FN: True,
                }

        tasks_to_attempt = experiment_state.get_tasks_for_ids(
            task_split=task_split, task_ids=task_batch_ids, include_samples=False
        )
        (
            new_frontiers,
            best_search_time_per_task,
        ) = self._neural_recognition_model.enumerateFrontiers(
            tasks=tasks_to_attempt,
            maximumFrontier=maximum_frontier,
            enumerationTimeout=enumeration_timeout,
            CPUs=cpus,
            solver=solver,
            evaluationTimeout=evaluation_timeout,
            max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
            solver_directory=solver_directory,
            testing=task_split == TEST,
        )

        # Re-score the frontiers under the grammar
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        new_frontiers = [grammar.rescoreFrontier(f) for f in new_frontiers]

        experiment_state.update_frontiers(
            new_frontiers=new_frontiers,
            maximum_frontier=maximum_frontier,
            task_split=task_split,
            is_sample=False,
        )

        experiment_state.best_search_times[task_split].update(best_search_time_per_task)

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
        cpus=os.cpu_count(),
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

        # Skip training if resume
        if experiment_state.metadata[INIT_FRONTIERS_FROM_CHECKPOINT]:
            if experiment_state.maybe_resume_from_checkpoint():
                print(
                    f"optimize_model_for_frontiers: Skipped recognition network training."
                )
                return {
                    SKIPPED_MODEL_FN: True,
                }

        # Initialize I/O example encoders.
        example_encoder = self._maybe_initialize_example_encoder(
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

    def _maybe_initialize_example_encoder(self, task_encoder_types, experiment_state):
        if model_loaders.EXAMPLES_ENCODER not in task_encoder_types:
            return None
        # Initialize from tasks.
        model_initializer_fn = experiment_state.models[model_loaders.EXAMPLES_ENCODER]
        return model_initializer_fn(experiment_state)
