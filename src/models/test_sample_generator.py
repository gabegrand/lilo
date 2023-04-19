"""
test_sample_generator.py.
"""
import src.models.sample_generator as to_test
from src.experiment_iterator import *
from src.models.gpt_base import DEFAULT_LINE_SEPARATOR, Prompt
from src.models.laps_grammar import LAPSGrammar
from src.task_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG


def get_initial_ground_truth_experiment_state(config):
    experiment_state = ExperimentState(config)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TEST)
    return experiment_state


def get_train_task_ids(experiment_state, n_task_ids=None):
    return {
        TRAIN: [
            t.name for t in list(experiment_state.task_frontiers[TRAIN])[:n_task_ids]
        ]
    }


def get_test_task_ids(experiment_state, n_task_ids=None):
    return {
        TEST: [t.name for t in list(experiment_state.task_frontiers[TEST])[:n_task_ids]]
    }


def get_sample_generator_and_state():
    config = TEST_GRAPHICS_CONFIG
    experiment_state = get_initial_ground_truth_experiment_state(config)
    sample_generator = to_test.GPTSampleGenerator()
    return sample_generator, experiment_state


def get_completion(
    experiment_state, sample_generator, prompt_text, n_samples_per_query
):
    completion, _ = sample_generator.get_completion_for_prompt(
        query_id=0,
        experiment_state=experiment_state,
        prompt_text=prompt_text,
        query_results_filepath=None,
        n_samples_per_query=n_samples_per_query,
        temperature=0.75,
        max_tokens=256,
        line_separator=DEFAULT_LINE_SEPARATOR,
        use_cached=False,
        debug=True,
    )
    return completion


def test_get_completion_for_prompt():
    sample_generator, experiment_state = get_sample_generator_and_state()
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    prompt = Prompt(
        experiment_state=experiment_state,
        body_task_ids=task_ids[TRAIN][:-1],
        final_task_id=task_ids[TRAIN][-1],
    )
    completion = get_completion(
        experiment_state, sample_generator, prompt.serialize(), n_samples_per_query
    )
    assert len(completion) > 0


def test_parse_completion():
    sample_generator, experiment_state = get_sample_generator_and_state()
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    prompt = Prompt(
        experiment_state=experiment_state,
        body_task_ids=task_ids[TRAIN][:-1],
        final_task_id=task_ids[TRAIN][-1],
    )
    completion = get_completion(
        experiment_state, sample_generator, prompt.serialize(), n_samples_per_query
    )
    parse_results = sample_generator.parse_completion(
        completion, experiment_state.models[model_loaders.GRAMMAR]
    )
    # Since we are using query_mock, all results should be valid.
    assert len(parse_results) == n_samples_per_query
    for result in parse_results:
        assert result["valid"]


def test_add_samples_to_experiment_state():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    prompt = Prompt(
        experiment_state=experiment_state,
        body_task_ids=task_ids[TRAIN][:-1],
        final_task_id=task_ids[TRAIN][-1],
    )
    completion = get_completion(
        experiment_state, sample_generator, prompt.serialize(), n_samples_per_query
    )
    parse_results = sample_generator.parse_completion(
        completion, experiment_state.models[model_loaders.GRAMMAR]
    )
    sample_generator.add_samples_to_experiment_state(
        experiment_state, TRAIN, parse_results
    )
    assert len(experiment_state.sample_tasks[TRAIN]) == n_samples_per_query
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    query_results = sample_generator.generate_samples(
        experiment_state=experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        n_samples=n_samples_per_query,
        body_task_types=[PROGRAMS],
        final_task_types=[PROGRAMS],
        final_task_origin=sample_generator.FINAL_TASK_ORIGIN_DEFAULT,
        function_name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        debug=True,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) == n_samples_per_query
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples_alternative_naming():
    sample_generator, experiment_state = get_sample_generator_and_state()
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    query_results = sample_generator.generate_samples(
        experiment_state=experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        n_samples=n_samples_per_query,
        body_task_types=[PROGRAMS],
        final_task_types=[PROGRAMS],
        final_task_origin=sample_generator.FINAL_TASK_ORIGIN_DEFAULT,
        function_name_classes=[
            LAPSGrammar.HUMAN_READABLE,
            LAPSGrammar.NUMERIC_FUNCTION_NAMES,
        ],
        debug=True,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) == n_samples_per_query
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples_with_language():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    query_results = sample_generator.generate_samples(
        experiment_state=experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        n_samples=n_samples_per_query,
        body_task_types=[PROGRAMS, LANGUAGE],
        final_task_types=[PROGRAMS],
        final_task_origin=sample_generator.FINAL_TASK_ORIGIN_DEFAULT,
        function_name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        debug=True,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) == n_samples_per_query
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty


def test_generate_samples_final_task_origin_train():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    query_results = sample_generator.generate_samples(
        experiment_state=experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        n_samples=n_samples_per_query,
        body_task_types=[PROGRAMS, LANGUAGE],
        final_task_types=[PROGRAMS],
        final_task_origin=sample_generator.FINAL_TASK_ORIGIN_RANDOM_TRAIN,
        function_name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        debug=True,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) == n_samples_per_query
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty

    final_task_id = query_results["results_by_query"][0]["prompt"]["final_task_data"][
        "task_id"
    ]
    assert final_task_id not in task_ids[TRAIN]
    assert final_task_id in get_train_task_ids(experiment_state)[TRAIN]


def test_generate_samples_final_task_origin_test():
    sample_generator, experiment_state = get_sample_generator_and_state()
    assert len(experiment_state.sample_tasks[TRAIN]) == 0
    task_ids = get_train_task_ids(experiment_state, n_task_ids=20)
    n_samples_per_query = 10
    query_results = sample_generator.generate_samples(
        experiment_state=experiment_state,
        task_splits=[TRAIN],
        task_ids_in_splits=task_ids,
        n_samples=n_samples_per_query,
        body_task_types=[PROGRAMS, LANGUAGE],
        final_task_types=[PROGRAMS],
        final_task_origin=sample_generator.FINAL_TASK_ORIGIN_RANDOM_TEST,
        function_name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        debug=True,
    )
    assert len(experiment_state.sample_tasks[TRAIN]) == n_samples_per_query
    for sample_task in experiment_state.sample_tasks[TRAIN]:
        assert not experiment_state.sample_frontiers[TRAIN][sample_task].empty

    final_task_id = query_results["results_by_query"][0]["prompt"]["final_task_data"][
        "task_id"
    ]
    assert final_task_id not in task_ids[TRAIN]
    assert final_task_id not in get_train_task_ids(experiment_state)[TRAIN]
    assert final_task_id in get_test_task_ids(experiment_state)[TEST]
