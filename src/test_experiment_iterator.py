"""
test_experiment_iterator.py | Author : Catherine Wong.

Usage: pytest -m src/test_experiment_iterator.py
"""
from data.compositional_graphics.encoder import *
from data.compositional_graphics.grammar import *
from data.compositional_graphics.make_tasks import *
from dreamcoder.frontier import Frontier
from src.experiment_iterator import *
from src.models.laps_dreamcoder_recognition import *
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import *
from src.task_loaders import *

grammar_config_block = {
    MODEL_TYPE: GRAMMAR,
    MODEL_LOADER: LogoGrammarLoader.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {},
}
example_encoder_config_block = {
    MODEL_TYPE: EXAMPLES_ENCODER,
    MODEL_LOADER: LogoFeatureCNNExamplesEncoder.name,
    MODEL_INITIALIZER_FN: "load_model_initializer",
    PARAMS: {"cuda": False},
}
amortized_synthesis_config_block = {
    MODEL_TYPE: AMORTIZED_SYNTHESIS,
    MODEL_LOADER: LAPSDreamCoderRecognitionLoader.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {},
}

TEST_GRAPHICS_CONFIG = {
    METADATA: {
        EXPERIMENT_ID: "dreamcoder_compositional_graphics_200_human",
        LOG_DIRECTORY: "experiments/logs/compositional_graphics",
        EXPORT_DIRECTORY: "experiments/outputs/compositional_graphics_test",
        RESUME_CHECKPOINT_DIRECTORY: "experiments/logs/compositional_graphics_test",
        EXPORT_WITH_TIMESTAMP: False,
        TASKS_LOADER: CompositionalGraphics200Loader.name,
        TASK_LANGUAGE_LOADER: CompositionalGraphics200HumanLanguageLoader.name,
        INIT_FRONTIERS_FROM_CHECKPOINT: False,
        OCAML_SPECIAL_HANDLER: OCAML_SPECIAL_HANDLER_LOGO,
        RANDOM_SEED: 0,
    },
    MODEL_INITIALIZERS: [
        grammar_config_block,
        example_encoder_config_block,
        amortized_synthesis_config_block,
    ],
    EXPERIMENT_ITERATOR: {
        MAX_ITERATIONS: 1,
        TASK_BATCHER: None,
        LOOP_BLOCKS: [
            # Enumerate from the grammar
            {
                EXPERIMENT_BLOCK_TYPE: EXPERIMENT_BLOCK_TYPE_MODEL_FN,
                MODEL_TYPE: GRAMMAR,
                EXPERIMENT_BLOCK_TYPE_MODEL_FN: "infer_programs_for_tasks",
                TASK_SPLIT: TRAIN,
                TASK_BATCH_SIZE: 2,
                PARAMS: {"enumeration_timeout": 10},
            }
        ],
    },
}

# Tests for ExperimentState
def test_init_tasks_from_config():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)
    for split in [TRAIN, TEST]:
        assert len(test_experiment_state.tasks[split]) > 0
        for task in test_experiment_state.tasks[split]:
            assert type(test_experiment_state.task_frontiers[split][task]) == Frontier


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
    assert type(test_experiment_state.models[GRAMMAR]) == LAPSGrammar


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


def test_get_tasks_with_samples():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    # Add a sample task in each.
    for split in (TRAIN, TEST):
        test_experiment_state.sample_tasks[split] += ["test"]

    for split in (TRAIN, TEST):
        all_tasks = test_experiment_state.get_tasks_for_ids(
            task_split=split, task_ids=ExperimentState.ALL, include_samples=True
        )
        all_tasks_no_samples = test_experiment_state.get_tasks_for_ids(
            task_split=split, task_ids=ExperimentState.ALL, include_samples=False
        )
        assert len(all_tasks) == len(all_tasks_no_samples) + 1


def test_get_language_for_ids():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_task_ids = ["a small triangle", "a medium triangle"]

    test_language = test_experiment_state.get_language_for_ids(
        task_split=TRAIN, task_ids=test_task_ids
    )

    assert len(test_task_ids) == len(test_language)
    for task_language in test_language:
        assert len(task_language) > 1

    # Check ALL
    for split in (TRAIN, TEST):
        all_language = test_experiment_state.get_language_for_ids(
            task_split=split, task_ids=ExperimentState.ALL
        )
        assert len(all_language) == len(test_experiment_state.tasks[split])


def test_get_language_for_ids_with_samples():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)
    # Check ALL
    for split in (TRAIN, TEST):
        all_language = test_experiment_state.get_language_for_ids(
            task_split=split, task_ids=ExperimentState.ALL, include_samples=True
        )
        assert len(all_language) == len(test_experiment_state.tasks[split])


def test_checkpoint_resume_frontiers():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_experiment_state.curr_iteration = 0

    test_experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    test_experiment_state.checkpoint_frontiers()

    test_experiment_state.reset_task_frontiers(task_split=TRAIN, task_ids=ALL)

    assert len(test_experiment_state.get_non_empty_frontiers_for_split(TRAIN)) == 0

    test_experiment_state.load_frontiers_from_checkpoint()

    assert len(test_experiment_state.get_non_empty_frontiers_for_split(TRAIN)) == len(
        test_experiment_state.task_frontiers[TRAIN]
    )
