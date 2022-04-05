"""
test_sample_generator.py.
"""
from random import sample
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


def get_train_task_ids(experiment_state, n_task_ids=20):
    return {
        TRAIN: [
            t.name for t in list(experiment_state.task_frontiers[TRAIN])[:n_task_ids]
        ]
    }


def get_sample_generator_and_state():
    config = TEST_GRAPHICS_CONFIG
    experiment_state = get_initial_ground_truth_experiment_state(config)
    sample_generator = to_test.CodexSampleGenerator()
    return sample_generator, experiment_state


def test_sample_prompt_training_examples_programs():
    sample_generator, experiment_state = get_sample_generator_and_state()

    # Test when we have more than enough possible tasks.
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_train_examples_per_prompt = 10

    sample_examples = sample_generator.sample_prompt_training_examples(
        experiment_state,
        n_train_examples_per_prompt=n_train_examples_per_prompt,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        prompt_example_types=[PROGRAMS],
    )
    assert len(sample_examples) == n_train_examples_per_prompt
    assert len(sample_examples[0]) == 1

    # Test when we have  enough possible tasks.
    max_task_ids = 5
    task_ids = get_train_task_ids(experiment_state, n_task_ids=max_task_ids)
    n_train_examples_per_prompt = 10

    sample_examples = sample_generator.sample_prompt_training_examples(
        experiment_state,
        n_train_examples_per_prompt=n_train_examples_per_prompt,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        prompt_example_types=[PROGRAMS],
    )
    assert len(sample_examples) == max_task_ids
    assert len(sample_examples[0]) == 1


def test_sample_prompt_training_examples_language():
    sample_generator, experiment_state = get_sample_generator_and_state()
    # Test when we have more than enough possible tasks.
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_train_examples_per_prompt = 10

    sample_examples = sample_generator.sample_prompt_training_examples(
        experiment_state,
        n_train_examples_per_prompt=n_train_examples_per_prompt,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        prompt_example_types=[LANGUAGE, PROGRAMS],
        allow_duplicate_examples_per_task=False,
        allow_language_for_disjoint_tasks=False,
    )
    assert len(sample_examples) == n_train_examples_per_prompt + 1
    assert len(sample_examples[0]) == 2
    assert len(sample_examples[-1]) == 1


def test_generate_codex_prompt_programs_only():
    sample_generator, experiment_state = get_sample_generator_and_state()
    # Test when we have more than enough possible tasks.
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_train_examples_per_prompt = 10
    _, prompt_text = sample_generator.generate_codex_prompt_text(
        experiment_state,
        n_train_examples_per_prompt=n_train_examples_per_prompt,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        prompt_example_types=[PROGRAMS],
        allow_duplicate_examples_per_task=False,
        allow_language_for_disjoint_tasks=False,
    )


def test_generate_codex_prompt_programs_language():
    sample_generator, experiment_state = get_sample_generator_and_state()
    # Test when we have more than enough possible tasks.
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_train_examples_per_prompt = 10
    _, prompt_text = sample_generator.generate_codex_prompt_text(
        experiment_state,
        n_train_examples_per_prompt=n_train_examples_per_prompt,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        prompt_example_types=[PROGRAMS, LANGUAGE],
        allow_duplicate_examples_per_task=False,
        allow_language_for_disjoint_tasks=False,
    )


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


def test_maybe_get_frontiers_from_completion():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0

    # Test the mock query.
    n_samples = 5
    mock_completion = sample_generator.query_mock(experiment_state, n_samples=n_samples)
    (
        valid_programs,
        invalid_programs,
    ) = sample_generator.maybe_get_frontiers_from_completion(
        experiment_state, mock_completion
    )
    assert len(valid_programs) == n_samples
    assert len(invalid_programs) == 0
    assert len(experiment_state.sample_tasks[TRAIN]) > 0
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples_programs_only():
    # Test mock sample generation
    sample_generator, experiment_state = get_sample_generator_and_state()
    example_task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    n_samples = 5
    sample_generator.generate_samples(
        experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=example_task_ids,
        n_samples=n_samples,
        n_samples_per_prompt=1,
        prompt_example_types=[PROGRAMS],
        debug=False,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) > 0
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples_programs_language():
    # Test mock sample generation
    sample_generator, experiment_state = get_sample_generator_and_state()
    example_task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    n_samples = 5
    sample_generator.generate_samples(
        experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=example_task_ids,
        n_samples=n_samples,
        n_samples_per_prompt=1,
        prompt_example_types=[PROGRAMS, LANGUAGE],
        debug=False,
        verbose_prompt=True,
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
        function_name_classes=[
            LAPSGrammar.HUMAN_READABLE,
            LAPSGrammar.NUMERIC_FUNCTION_NAMES,
        ],
    )
    assert len(experiment_state.sample_tasks[TRAIN]) > 0
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty
