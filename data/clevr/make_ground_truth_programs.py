"""
make_ground_truth_programs.py | Author : Catherine Wong.
"""

import re
from dreamcoder.program import Primitive, Program
import dreamcoder.domains.clevr.clevrPrimitives as clevrPrimitives

from re import *

MAKE_GT_PROGRAMS_REGISTRY = dict()

# Types of CLEVR questions
LOCALIZATION = "2_localization"
COMPARE_INTEGER = "1_compare_integer"
ZERO_HOP = "1_zero_hop"


def register(name):
    def wrapper(f):
        MAKE_GT_PROGRAMS_REGISTRY[name] = f
        return f

    return wrapper


def make_ground_truth_program_for_task(t):
    task_type, task_language = get_task_type_language(t)
    ground_truth_program_fn = MAKE_GT_PROGRAMS_REGISTRY[task_type]
    ground_truth_program = ground_truth_program_fn(task_language, t)
    check_task_evaluation(t, ground_truth_program, should_succeed=True)
    # TODO: make them canonical Eta-Long form?
    return ground_truth_program


def get_task_type_language(t):
    split_name = t.name.split("-")
    task_type, task_language = split_name[1], split_name[-1]
    return task_type, task_language


def check_task_evaluation(test_task, raw_program, should_succeed=True):
    clevrPrimitives.clevr_original_v1_primitives()
    print(f"Testing program: {raw_program}")
    p = Program.parse(raw_program)
    test_pass = test_task.check(p, timeout=1000)
    print(f"{test_task.name} | pass: {test_pass}")
    assert test_pass == should_succeed


def get_filter_conditions_from_string(filter_string):
    """
    Takes a question string of the form "brown cubes"; "big brown"
    Builds a list of string filter conditions of the form:
        (clevr_eq_{TYPE} clevr_{VALUE} (clevr_query_{TYPE} $1))
    eg. (clevr_eq_size clevr_small (clevr_query_size $1))
    """
    filter_conditions = []
    filter_strings = filter_string.split()
    for filter_str_condition in filter_strings:
        if "thing" in filter_str_condition or "object" in filter_str_condition:
            continue
        filter_value, filter_type = get_filter_value_type_from_string(
            filter_str_condition
        )
        filter_conditions.append(
            f"(clevr_eq_{filter_type} clevr_{filter_value} (clevr_query_{filter_type} $1))"
        )
    return filter_conditions


def get_filter_value_type_from_string(filter_str_condition):
    if filter_str_condition[-1] == "s":
        filter_str_condition = filter_str_condition[:-1]
    synonyms = {
        "thing": ["thing", "object"],
        "sphere": ["sphere", "ball"],
        "cube": ["cube", "block"],
        "large": ["large", "big"],
        "small": ["small", "tiny"],
        "metal": ["metallic", "metal", "shiny"],
        "rubber": ["rubber", "matte"],
        "left": ["left of", "to the left of", "on the left side of"],
        "right": ["right of", "to the right of", "on the right side of"],
        "behind": ["behind"],
        "front": ["in front of"],
        "above": ["above"],
        "below": ["below"],
    }
    types_to_values = {
        "shape": ["cube", "sphere", "cylinder"],
        "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
        "relation": ["left", "right", "behind", "front"],
        "size": ["small", "large"],
        "material": ["rubber", "metal"],
    }
    value = filter_str_condition
    for canonical_value in synonyms:
        if filter_str_condition in synonyms[canonical_value]:
            value = canonical_value
            break

    for types in types_to_values:
        if value in types_to_values[types]:
            return value, types
    assert False


def make_filter_program(filter_conditions, is_terminal=False):
    filter_str = None
    for condition in filter_conditions:
        fold_fn = f"(lambda (lambda (clevr_if {condition} (clevr_add $1 $0) $0)))"
        if filter_str is None:
            filter_str = f"(clevr_fold $0 clevr_empty {fold_fn})"
        else:
            filter_str = f"(clevr_fold {filter_str} clevr_empty {fold_fn})"
    if is_terminal:
        filter_str = f"(lambda {filter_str} )"
    return filter_str


def make_count_program(filter_conditions, is_terminal=False):
    filter_str = make_filter_program(filter_conditions, is_terminal=False)
    count_str = f"(clevr_count {filter_str})"
    if is_terminal:
        count_str = f"(lambda {count_str} )"
    return count_str


@register(LOCALIZATION)
def make_evaluate_ground_truth_program_localization(task_language, task):
    """Tasks of the form Find the XXX things."""
    prefix = "Find the "
    filter_string = task_language.replace(prefix, "").replace(".", "")

    filter_conditions = get_filter_conditions_from_string(filter_string)
    return make_filter_program(filter_conditions, is_terminal=True)


@register(COMPARE_INTEGER)
def make_evaluate_ground_truth_program_compare_integer(task_language, task):
    regexes = [
        (
            r"Is the number of (?P<first_count>.+) less than the number of (?P<second_count>.+)\?",
            "clevr_lt?",
        ),
        (
            r"Is the number of (?P<first_count>.+) greater than the number of (?P<second_count>.+)\?",
            "clevr_gt?",
        ),
        (
            r"Are there more (?P<first_count>.+) than (?P<second_count>.+)\?",
            "clevr_gt?",
        ),
        (
            r"Are there fewer (?P<first_count>.+) than (?P<second_count>.+)\?",
            "clevr_lt?",
        ),
    ]
    for regex, comparative_operator in regexes:
        match = re.match(regex, task_language)
        if match is not None:
            first_count, second_count = (
                match.group("first_count"),
                match.group("second_count"),
            )
            first_count, second_count = (
                make_count_program(get_filter_conditions_from_string(first_count)),
                make_count_program(get_filter_conditions_from_string(second_count)),
            )
            raw_program = (
                f"(lambda ({comparative_operator} {first_count} {second_count}))"
            )
            return raw_program
    assert False
