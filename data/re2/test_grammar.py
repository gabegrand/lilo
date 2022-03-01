"""
test_grammar.py
"""
from src.task_loaders import *
import data.re2.make_tasks as re2_tasks
import data.re2.grammar as to_test
from dreamcoder.program import Program

replace_sample_tasks = [
    (
        "re2_train_1_if_there_is_consonant_replace_that_with_j_p",
        ["consonant"],
        ["j", "p"],
    ),
    (
        "re2_train_125_if_there_is_b_any_letter_replace_that_with_n_c",
        ["b", "_rdot"],
        ["n", "c"],
    ),
    (
        "re2_train_10_if_there_is_any_letter_any_letter_replace_that_with_i",
        ["_rdot", "_rdot"],
        ["i"],
    ),
    (
        "re2_train_96_if_there_is_vowel_vowel_replace_that_with_y",
        ["vowel", "vowel"],
        ["y"],
    ),
]
starts_ends_replace_tasks = [
    (
        "re2_train_2_if_the_word_ends_with_consonant_s_replace_that_with_b",
        ["consonant", "s"],
        ["b"],
    ),
    (
        "re2_train_13_if_the_word_starts_with_vowel_any_letter_replace_that_with_q_v",
        ["vowel", "_rdot"],
        ["q", "v"],
    ),
    (
        "re2_train_36_if_the_word_starts_with_consonant_replace_that_with_p_i",
        ["consonant"],
        ["p", "i"],
    ),
]

prepend_postpend_remove_double_tasks = [
    ("re2_train_4_if_there_is_b_add_k_before_that", ["b"], ["k"]),
    (
        "re2_train_9_if_the_word_starts_with_any_letter_vowel_add_j_before_that",
        ["_rdot", "vowel"],
        ["j"],
    ),
    (
        "re2_train_34_if_the_word_ends_with_any_letter_add_d_after_that",
        ["_rdot"],
        ["d"],
    ),
    (
        "re2_train_379_if_there_is_t_any_letter_remove_that",
        ["t", "_rdot"],
        ["t", "_rdot"],
    ),
    (
        "re2_train_467_if_there_is_any_letter_any_letter_double_that",
        ["_rdot", "_rdot"],
        ["_rdot", "_rdot"],
    ),
    (
        "re2_train_5_if_the_word_ends_with_consonant_consonant_double_that",
        ["consonant", "consonant"],
        ["consonant", "consonant"],
    ),
]


def check_equal(name, raw, input, gold, task):
    p = raw
    output = p.evaluate([])(input)
    # Assert that this types correctly.
    assert p.infer() == task.request
    grammar = to_test.Re2GrammarLoader().load_model(experiment_state=None)
    try:
        grammar.logLikelihood(p.infer(), p)
    except:
        import pdb

        pdb.set_trace()
    print(f"{name} in: {input} | out: {output} | gold: {gold}")
    assert output == gold


def assert_regex_on_examples(raw_regex_program, task):
    for (input, output) in task.examples:
        check_equal(
            task.name, raw=raw_regex_program, input=input[0], gold=output, task=task
        )


def _get_sample_tasks(ids, split="train"):
    task_loader = TaskLoaderRegistry[re2_tasks.Re2Loader.name]
    tasks = task_loader.load_tasks()
    id_to_task = {t.name: t for t in tasks[split]}
    return [id_to_task[id] for id in ids]


def _get_language_from_name(task_name):
    return " ".join(task_name.split("_")[3:])


def test_synthetic_language_to_re2_program_get_match_type():
    task_ids = [t[0] for t in replace_sample_tasks]
    tasks = _get_sample_tasks(task_ids)
    assert len(tasks) == len(replace_sample_tasks)
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        match_type, match_tokens = to_test.get_match_type(language)
        assert match_type == to_test.ANYWHERE
        assert match_tokens == replace_sample_tasks[idx][1]


def test_synthetic_language_to_re2_program_get_match_type_prepend_postpend():
    task_ids = [t[0] for t in prepend_postpend_remove_double_tasks]
    tasks = _get_sample_tasks(task_ids)
    assert len(tasks) == len(prepend_postpend_remove_double_tasks)
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        match_type, match_tokens = to_test.get_match_type(language)
        assert match_tokens == prepend_postpend_remove_double_tasks[idx][1]
        replacement_type, replacement_tokens = to_test.get_replacement_type(
            language, match_tokens
        )
        assert replacement_tokens == prepend_postpend_remove_double_tasks[idx][-1]


def test_re2_legacy_programs():
    raw = "(lambda  (if (_rmatch (_ror _a _e) $0) _f $0)   )"
    check_equal("replace a -> f", raw, "a", "f")


def test_synthetic_language_to_re2_program_match_anywhere_replace():
    task_ids = [t[0] for t in replace_sample_tasks]
    print(task_ids)
    tasks = _get_sample_tasks(task_ids)
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        match_type, match_tokens = to_test.get_match_type(language)
        replacement_type, replacement_tokens = to_test.get_replacement_type(
            language, match_tokens
        )
        assert match_type == to_test.ANYWHERE
        assert match_tokens == replace_sample_tasks[idx][1]
        assert replacement_type == to_test.REPLACE_MATCH
        assert replacement_tokens == replace_sample_tasks[idx][-1]

        replace_regex = to_test.build_replace_regex(
            replacement_type, match_tokens, replacement_tokens
        )
        full_regex = to_test.build_match_replace_regex(
            match_type, replace_regex, match_tokens
        )
        assert_regex_on_examples(full_regex, t)


def test_synthetic_language_to_re2_program_starts_end_replace():
    task_ids = [t[0] for t in starts_ends_replace_tasks]
    print(task_ids)
    tasks = _get_sample_tasks(task_ids)
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        full_regex = to_test.synthetic_language_to_re2_program(language)
        assert_regex_on_examples(full_regex, t)


def test_synthetic_language_to_re2_program_prepend_postpend_remove():
    task_ids = [t[0] for t in prepend_postpend_remove_double_tasks]
    print(task_ids)
    tasks = _get_sample_tasks(task_ids)
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        full_regex = to_test.synthetic_language_to_re2_program(language)
        assert_regex_on_examples(full_regex, t)

