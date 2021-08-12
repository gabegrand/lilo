"""
test_experiment_iterator.py | Author : Catherine Wong.

Usage: pytest -m src/test_experiment_iterator.py
"""
from src.experiment_iterator import *

from data.compositional_graphics.make_tasks import *
from data.compositional_graphics.grammar import *

from dreamcoder.frontier import Frontier
from dreamcoder.grammar import Grammar

TEST_GRAPHICS_CONFIG = {
    METADATA: {
        TASKS_LOADER: CompositionalGraphics200Loader.name,
        TASK_LANGUAGE_LOADER: CompositionalGraphics200HumanLanguageLoader.name,
        INIT_FRONTIERS_FROM_CHECKPOINT: False,
        OCAML_SPECIAL_HANDLER: OCAML_SPECIAL_HANDLER_LOGO,
    },
    MODEL_INITIALIZERS: [
        {
            MODEL_TYPE: GRAMMAR,
            MODEL_LOADER: LogoGrammarLoader.name,
            MODEL_INITIALIZER_FN: "load_model",
            PARAMS: {},
        }
    ],
}

# Tests for ExperimentState
def test_init_tasks_from_config():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)
    for split in [TRAIN, TEST]:
        assert len(test_experiment_state.tasks[split]) > 0
        for task in test_experiment_state.tasks[split]:
            assert (
                type(test_experiment_state.task_frontiers[split][task]) == Frontier
            )


def test_init_task_language_from_config():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)
    for split in [TRAIN, TEST]:
        assert len(test_experiment_state.tasks[split]) > 0
        for task in test_experiment_state.tasks[split]:
            assert task.name in test_experiment_state.task_language[split]


def test_init_models_from_config():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    assert GRAMMAR in test_experiment_state.models
    assert type(test_experiment_state.models[GRAMMAR]) == Grammar


def test_get_tasks_for_ids():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_task_ids = ["a small triangle", "a medium triangle"]

    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split=TRAIN, task_ids=test_task_ids
    )

    assert len(test_tasks) == len(test_task_ids)
    for t in test_tasks:
        assert t.name in test_task_ids

    # Check ALL
    for split in (TRAIN, TEST):
        all_tasks = test_experiment_state.get_tasks_for_ids(
            task_split=split, task_ids=ExperimentState.ALL
        )
        assert len(all_tasks) == len(test_experiment_state.tasks[split])
