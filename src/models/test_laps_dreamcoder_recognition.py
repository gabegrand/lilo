"""
test_laps_dreamcoder_recognition.py | Author : Catherine Wong
"""
from src.task_loaders import *
from src.models.model_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState


def test_laps_dreamcoder_recognition_integration():
    """Note: this is an integration test that runs the training for a set time, then tries to enumerate from the model for search."""
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

    TEST_ENUMERATION_TIMEOUT = 10
    test_batch_ids = ["a small triangle", "a medium triangle"]
    test_recognition.infer_programs_for_tasks(
        test_experiment_state,
        task_split=TRAIN,
        task_batch_ids=test_batch_ids,
        enumeration_timeout=TEST_ENUMERATION_TIMEOUT,
    )

    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split=TRAIN, task_ids=test_batch_ids
    )
