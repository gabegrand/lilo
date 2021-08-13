"""
test_laps_grammar.py | Author : Catherine Wong
"""
from src.experiment_iterator import *
from src.task_loaders import *
from src.models.model_loaders import GRAMMAR
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG


def test_laps_grammar_infer_programs_for_tasks():
    """Note: this is an integration test that runs enumeration for a set time."""
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]

    TEST_ENUMERATION_TIMEOUT = 10

    test_batch_ids = ["a small triangle", "a medium triangle"]
    test_grammar.infer_programs_for_tasks(
        test_experiment_state,
        task_split=TRAIN,
        task_batch_ids=test_batch_ids,
        enumeration_timeout=TEST_ENUMERATION_TIMEOUT,
    )

    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split=TRAIN, task_ids=test_batch_ids
    )

    # At least one shouldn't be empty
    is_not_empty = False
    for task in test_tasks:
        if not test_experiment_state.task_frontiers[TRAIN][task].empty:
            is_not_empty = True

    assert is_not_empty


def test_laps_grammar_generative_sample_frontiers_for_tasks():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]

    TEST_ENUMERATION_TIMEOUT = 10
    test_grammar.generative_sample_frontiers_for_tasks(
        experiment_state=test_experiment_state,
        enumeration_timeout=TEST_ENUMERATION_TIMEOUT,
    )

    assert len(test_experiment_state.sample_frontiers) > 0
    for frontier_task in test_experiment_state.sample_frontiers:
        assert not test_experiment_state.sample_frontiers[frontier_task].empty


def test_laps_grammar_optimize_grammar_frontiers_for_frontiers():
    """Note: this is an integration test that runs enumeration for a set time."""
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]

    TEST_ENUMERATION_TIMEOUT = 10

    test_batch_ids = ["a small triangle", "a medium triangle"]
    test_grammar.infer_programs_for_tasks(
        experiment_state=test_experiment_state,
        task_split=TRAIN,
        task_batch_ids=test_batch_ids,
        enumeration_timeout=TEST_ENUMERATION_TIMEOUT,
    )

    pre_compression_grammar_type = type(test_experiment_state.models[GRAMMAR])

    test_grammar.optimize_grammar_frontiers_for_frontiers(
        experiment_state=test_experiment_state,
        task_split=TRAIN,
        task_batch_ids=ExperimentState.ALL,
    )

    assert (
        type(test_experiment_state.models[GRAMMAR]) == pre_compression_grammar_type
    )
