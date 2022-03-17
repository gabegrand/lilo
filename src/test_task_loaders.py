"""
test_task_loaders.py | Author : Catherine Wong
Utility classes for loading and batching datasets of tasks and language.
"""
from src.task_loaders import *

from src.experiment_iterator import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG


TEST_BATCH_SIZE = 15


def test_ordered_task_batcher():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    task_batcher = OrderedTaskBatcher(
        test_experiment_state, curr_iteration=None, max_iterations=None
    )


def test_ground_truth_ordered_task_batcher():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    task_batcher = GroundTruthOrderedTaskBatcher(
        test_experiment_state,
        curr_iteration=None,
        max_iterations=None,
        global_batch_size=TEST_BATCH_SIZE,
    )

    for split in task_batcher.task_id_orderings:
        assert len(task_batcher.task_id_orderings[split]) == len(
            test_experiment_state.task_frontiers[split]
        )

    tasks = task_batcher.get_task_batch_ids(
        test_experiment_state, curr_iteration=0, task_split="train", batch_size=ALL
    )
    assert len(tasks) == 199

    tasks = task_batcher.get_task_batch_ids(
        test_experiment_state,
        curr_iteration=0,
        task_split="train",
        batch_size=GLOBAL_BATCH_SIZE,
    )
    assert len(tasks) == TEST_BATCH_SIZE

    tasks_2 = task_batcher.get_task_batch_ids(
        test_experiment_state,
        curr_iteration=0,
        task_split="train",
        batch_size=GLOBAL_BATCH_SIZE,
    )
    assert tasks[0] == tasks_2[0]

    tasks_3 = task_batcher.get_task_batch_ids(
        test_experiment_state, curr_iteration=0, task_split="train", batch_size=15,
    )
    assert tasks[0] == tasks_3[0]
    assert len(tasks_3) == 15

