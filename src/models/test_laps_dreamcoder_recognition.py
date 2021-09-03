"""
test_laps_dreamcoder_recognition.py | Author : Catherine Wong
"""
from src.task_loaders import *
from src.models.model_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState


def test_laps_dreamcoder_recognition_optimize_model_for_frontiers_no_samples():
    """Note: this is an integration test that runs the training for a set time."""
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_recognition = test_experiment_state.models[AMORTIZED_SYNTHESIS]

    task_encoder_types = [EXAMPLES_ENCODER]
    recognition_train_steps = 10

    test_recognition.optimize_model_for_frontiers(
        experiment_state=test_experiment_state,
        task_split=TRAIN,
        task_batch_ids=ALL,
        recognition_train_steps=recognition_train_steps,
    )
