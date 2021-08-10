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
