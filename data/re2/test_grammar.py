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
        "re2_train_10_if_there_is_any_letter_any_letter_replace_that_with_i",
        [".", "."],
        ["i"],
    ),
    (
        "re2_train_96_if_there_is_vowel_vowel_replace_that_with_y",
        ["vowel", "vowel"],
        ["y"],
    ),
    (
        "re2_train_125_if_there_is_b_any_letter_replace_that_with_n_c",
        ["b", "."],
        ["n", "c"],
    ),
]


def check_equal(name, raw, input, gold):
    p = Program.parse(raw)
    output = p.evaluate([])(input)
    print(f"{name} in: {input} | out: {output} | gold: {gold}")
    assert output == gold


def assert_regex_on_examples(raw_regex_program, task):
    for (input, output) in task.examples:
        import pdb

        pdb.set_trace()


def _get_sample_tasks(ids, split="train"):
    task_loader = TaskLoaderRegistry[re2_tasks.Re2Loader.name]
    tasks = task_loader.load_tasks()
    return [t for t in tasks[split] if t.name in ids]


def _get_language_from_name(task_name):
    return " ".join(task_name.split("_")[3:])


def test_synthetic_language_to_re2_program_match_start():
    pass


def test_synthetic_language_to_re2_program_match_end():
    pass


def test_synthetic_language_to_re2_program_match_anywhere_replace():
    tasks = _get_sample_tasks([t[0] for t in replace_sample_tasks])
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
        full_regex = to_test.build_full_regex(match_type, replace_regex, match_tokens)
        assert_regex_on_examples(full_regex, t)


def test_synthetic_language_to_re2_program_match_anywhere():
    task_loader = TaskLoaderRegistry[re2_tasks.Re2Loader.name]
    tasks = task_loader.load_tasks()
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        match_type, match_tokens = to_test.get_match_type(language)

        if match_type == to_test.ANYWHERE:
            pass
            # If replace.

            #


def test_synthetic_language_to_re2_program_get_match_type():
    tasks = _get_sample_tasks(replace_sample_tasks)
    for idx, t in enumerate(tasks):
        language = _get_language_from_name(t.name)
        match_type, match_tokens = to_test.get_match_type(language)
        assert match_type == to_test.ANYWHERE
        assert match_tokens == replace_sample_tasks[idx][1]
