"""
test_laps_grammar.py | Author : Catherine Wong
"""
from src.experiment_iterator import *
from src.task_loaders import *
from src.models.model_loaders import GRAMMAR
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG
from dreamcoder.program import Program


def test_laps_grammar_show_alternate_primitives_default():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]
    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split="train", task_ids="all"
    )
    for t in test_tasks:
        p = t.groundTruthProgram
        for name_class in [
            test_grammar.NUMERIC_FUNCTION_NAMES,
            test_grammar.HUMAN_READABLE,
        ]:
            alternate_name = test_grammar.show_program(p, name_classes=[name_class])
            print(name_class)
            print(alternate_name)
            invert_alternate_naming = test_grammar.show_program(
                alternate_name, input_name_class=[name_class]
            )
            assert alternate_name != str(p)
            assert invert_alternate_naming == str(p)


def test_laps_grammar_ast_primitives():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]
    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split="train", task_ids="all"
    )
    for t in test_tasks:
        p = t.groundTruthProgram
        for name_class in [
            test_grammar.NUMERIC_FUNCTION_NAMES,
            test_grammar.HUMAN_READABLE,
        ]:
            alternate_name = test_grammar.show_program(p, name_classes=[name_class])
            print(name_class)
            print(alternate_name)
            invert_alternate_naming = test_grammar.show_program(
                alternate_name, input_name_class=[name_class]
            )
            assert alternate_name != str(p)
            assert invert_alternate_naming == str(p)


def test_laps_grammar_show_alternate_primitives():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]

    # Set the primitives to alternate human readable names.
    for p in test_grammar.primitives:
        test_grammar.set_function_name(
            str(p), name_class=test_grammar.HUMAN_READABLE, name=p.alternate_names[-1]
        )
    # Test program.
    test_batch_ids = ["a small triangle", "a medium triangle"]

    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split="train", task_ids=test_batch_ids
    )
    for t in test_tasks:
        test_program_no_inventions = t.groundTruthProgram
        new_name = test_grammar.show_program(
            test_program_no_inventions, name_classes=[test_grammar.HUMAN_READABLE]
        )
        print(f"Original: {str(test_program_no_inventions)}")
        print(new_name)

        new_str_name = test_grammar.show_program(
            str(test_program_no_inventions), name_classes=[test_grammar.HUMAN_READABLE]
        )
        assert new_str_name == new_name

    # Test some programs with inventions.
    dc_body = "#(lambda (lambda (logo_forLoop $1 (lambda (lambda (logo_FWRT (logo_MULL logo_UL $2) (logo_DIVA logo_UA $3) $0))))))"
    stitch_name_class, stitch_name = "stitch", "inv0"
    stitch_lam = "lam"
    test_grammar.function_names[dc_body] = dict(
        {test_grammar.DEFAULT_FUNCTION_NAMES: dc_body}
    )
    test_grammar.set_function_name(
        production_key=dc_body, name_class=stitch_name_class, name=stitch_name,
    )
    # Should now be written as DreamCoder
    test_program = "(lam (inv0 9 1 (logo_PT (lam (logo_FWRT (logo_MULL logo_UL 2) logo_ZA $0)) (inv0 4 2 $0))))".replace(
        stitch_lam, "lambda"
    ).replace(
        stitch_name, dc_body
    )
    test_program = Program.parse(test_program)
    new_name = test_grammar.show_program(
        test_program, name_classes=[stitch_name_class], lam=stitch_lam
    )
    print(f"Original: {str(test_program)}")
    print(new_name)

    new_str_name = test_grammar.show_program(
        str(test_program), name_classes=[stitch_name_class], lam=stitch_lam
    )
    assert new_str_name == new_name

    # Finally, test going the other way.
    original_dc = test_grammar.show_program(
        new_name, input_lam=stitch_lam, input_name_class=[stitch_name_class]
    )
    assert original_dc == str(test_program)


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

    assert type(test_experiment_state.models[GRAMMAR]) == pre_compression_grammar_type


def test_laps_grammar_send_receive_compressor_api_call():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]

    test_frontiers = test_experiment_state.get_frontiers_for_ids_in_splits(
        task_splits=[task_loaders.TRAIN, task_loaders.TEST],
        task_ids_in_splits={
            task_loaders.TRAIN: ["a small triangle", "a medium triangle"],
            task_loaders.TEST: [],
        },
    )

    (
        json_response,
        json_error,
        json_serialized_binary_message,
    ) = test_grammar._send_receive_compressor_api_call(
        api_fn=test_grammar.TEST_API_FN,
        grammar=None,
        frontiers=test_frontiers,
        kwargs={"test_int_kwarg": 1},
    )

    assert (
        json_serialized_binary_message[test_grammar.API_FN] == test_grammar.TEST_API_FN
    )
    deserialized_grammar = json_response[test_grammar.REQUIRED_ARGS][GRAMMAR][0]
    assert len(deserialized_grammar) == len(test_grammar)

    for split in [TRAIN, TEST]:
        assert (
            len(
                json_response[test_grammar.REQUIRED_ARGS][test_grammar.FRONTIERS][split]
            )
            == 0
        )


def test_laps_grammar_checkpoint_reload(tmpdir):
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_grammar = test_experiment_state.models[GRAMMAR]

    test_grammar.checkpoint(test_experiment_state, tmpdir)

    new_grammar = test_grammar.load_model_from_checkpoint(test_experiment_state, tmpdir)

    assert test_grammar == new_grammar
