"""
test_sample_generator.py.
"""
import src.models.sample_generator as to_test
from src.experiment_iterator import *
from src.task_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG
from src.models.laps_grammar import LAPSGrammar


def get_initial_ground_truth_experiment_state(config):
    experiment_state = ExperimentState(config)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TEST)
    return experiment_state


def get_sample_generator_and_state():
    config = TEST_GRAPHICS_CONFIG
    experiment_state = get_initial_ground_truth_experiment_state(config)
    sample_generator = to_test.CodexSampleGenerator()
    return sample_generator, experiment_state


def test_query_codex():
    sample_generator, experiment_state = get_sample_generator_and_state()
    test_prompt = ["function_1", "function_2", "function_3"]
    test_prompt = (
        sample_generator.DEFAULT_SEPARATOR.join(test_prompt)
        + sample_generator.DEFAULT_SEPARATOR
    )
    test_n_samples = 2
    completion = sample_generator.query_codex(
        prompt=test_prompt, n_samples=test_n_samples
    )
    assert len(completion) > 0


def test_generate_samples():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    n_samples = 5
    sample_generator.generate_samples(
        experiment_state, task_splits=None, task_ids_in_splits=None, n_samples=n_samples
    )
    assert len(experiment_state.sample_tasks[TRAIN]) > 0
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples_alternate_naming():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    n_samples = 5
    sample_generator.generate_samples(
        experiment_state,
        task_splits=None,
        task_ids_in_splits=None,
        n_samples=n_samples,
        function_name_class=LAPSGrammar.NUMERIC_FUNCTION_NAMES,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) > 0
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty
