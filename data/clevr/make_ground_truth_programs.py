"""
make_ground_truth_programs.py | Author : Catherine Wong.
"""

import re
from dreamcoder.program import EtaLongVisitor, Program
import dreamcoder.domains.clevr.clevrPrimitives as clevrPrimitives

from re import *

MAKE_GT_PROGRAMS_REGISTRY = dict()

# Types of CLEVR questions
LOCALIZATION = "2_localization"
COMPARE_INTEGER = "1_compare_integer"
ZERO_HOP = "1_zero_hop"
REMOVE = "2_remove"
TRANSFORM = "2_transform"
SINGLE_OR = "1_single_or"


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
    ground_truth_program = Program.parse(ground_truth_program)
    assert ground_truth_program.infer() == t.request
    ground_truth_program = EtaLongVisitor(request=t.request).execute(
        ground_truth_program
    )
    return ground_truth_program


def get_task_type_language(t):
    split_name = t.name.split("-")
    task_type, task_language = split_name[1], split_name[-1]
    return task_type, task_language


def check_task_evaluation(test_task, raw_program, should_succeed=True):
    clevrPrimitives.clevr_original_v1_primitives()
    clevrPrimitives.clevr_map_transform_primitives()
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


def get_map_transform_from_string(initial_filter, map_transform_string):
    map_string = None
    filter_strings = map_transform_string.split()

    for filter_str_condition in filter_strings:
        if "thing" in filter_str_condition or "object" in filter_str_condition:
            continue
        filter_value, filter_type = get_filter_value_type_from_string(
            filter_str_condition
        )
        if map_string is None:
            map_string = initial_filter

        map_string = f"(clevr_map (clevr_transform_{filter_value} clevr_{filter_type}) {map_string})"

    return map_string


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
    import pdb

    pdb.set_trace()
    assert False


def make_filter_program(filter_conditions, with_respect_to="$0", is_terminal=False):
    filter_str = None
    for condition in filter_conditions:
        fold_fn = f"(lambda (lambda (clevr_if {condition} (clevr_add $1 $0) $0)))"
        if filter_str is None:
            filter_str = f"(clevr_fold {with_respect_to} clevr_empty {fold_fn})"
        else:
            filter_str = f"(clevr_fold {filter_str} clevr_empty {fold_fn})"
    if is_terminal:
        filter_str = f"(lambda {filter_str} )"
    return filter_str


def make_count_program(filter_conditions, with_respect_to="$0", is_terminal=False):
    filter_str = make_filter_program(
        filter_conditions, with_respect_to, is_terminal=False
    )
    if filter_str is None:
        filter_str = with_respect_to
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


@register(TRANSFORM)
def make_evaluate_ground_truth_program_transform(task_language, task):
    # TODO: under construction
    TRANSFORM_ONLY, COUNT = "transform_only", "count"
    regexes = [
        (
            r"What if all the (?P<first_filter>.+) became (?P<second_filter>.+)\?",
            TRANSFORM_ONLY,
        ),
        (
            r"What if the (?P<first_filter>.+) became a (?P<second_filter>.+)\?",
            TRANSFORM_ONLY,
        ),
        (
            r"If all of the (?P<first_filter>.+) became (?P<second_filter>.+), how many (?P<third_filter>.+) would there be\?",
            COUNT,
        ),
    ]
    for regex, question_type in regexes:
        match = re.match(regex, task_language)
        if match is not None:
            first_filter, second_filter = (
                match.group("first_filter"),
                match.group("second_filter"),
            )

            first_filter_str = make_filter_program(
                get_filter_conditions_from_string(first_filter)
            )
            map_transform_str = get_map_transform_from_string(
                first_filter_str, second_filter
            )
            if question_type == TRANSFORM_ONLY:
                first_filter = match.group("first_filter")
                return f"(lambda (clevr_union {map_transform_str} $0))"
            elif question_type == COUNT:
                third_filter = (match.group("third_filter"),)
                union = f"(clevr_union {map_transform_str} $0)"
                final_filter = make_filter_program(
                    get_filter_conditions_from_string(third_filter),
                    with_respect_to=union,
                    is_terminal=False,
                )
                return f"(lambda (clevr_count {final_filter}))"
            else:
                assert False
    assert False


@register(REMOVE)
def make_evaluate_ground_truth_program_remove(task_language, task):
    REMOVE_ONLY, COUNT = "remove_only", "count"
    regexes = [
        (r"What if you removed all of the (?P<first_filter>.+)\?", REMOVE_ONLY),
        (
            r"If you removed the (?P<first_filter>.+), how many (?P<second_filter>.+) would be left\?",
            COUNT,
        ),
    ]
    for regex, question_type in regexes:
        match = re.match(regex, task_language)
        if match is not None:
            if question_type == REMOVE_ONLY:
                first_filter = match.group("first_filter")
                filter_str = make_filter_program(
                    get_filter_conditions_from_string(first_filter)
                )
                raw_program = f"(lambda (clevr_difference $0 {filter_str}))"
            elif question_type == COUNT:

                first_filter, second_filter = (
                    match.group("first_filter"),
                    match.group("second_filter"),
                )
                first_filter_str, second_filter_str = (
                    make_filter_program(
                        get_filter_conditions_from_string(first_filter)
                    ),
                    make_filter_program(
                        get_filter_conditions_from_string(second_filter)
                    ),
                )
                if second_filter_str is None:
                    second_filter_str = "$0"
                raw_program = (
                    f"(clevr_difference {second_filter_str} {first_filter_str})"
                )

                raw_program = f"(lambda (clevr_count {raw_program}))"
            else:
                assert False
            return raw_program
    assert False


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


@register(SINGLE_OR)
def make_evaluate_ground_truth_program_single_or(task_language, task):
    task_language = task_language.replace(" either", "")
    regexes = [
        r"What number of (?P<first_count>.+) are (?P<first_or>.+) or (?P<second_or>.+)\?",
        r"How many (?P<first_count>.+) are (?P<first_or>.+) or (?P<second_or>.+)\?",
    ]
    for regex in regexes:
        match = re.match(regex, task_language)
        if match is not None:
            first_count, first_or, second_or = (
                match.group("first_count"),
                match.group("first_or"),
                match.group("second_or"),
            )
            first_or, second_or = (
                make_filter_program(get_filter_conditions_from_string(first_or)),
                make_filter_program(get_filter_conditions_from_string(second_or)),
            )

            union = f"(clevr_union {first_or} {second_or})"
            return make_count_program(
                get_filter_conditions_from_string(first_count),
                with_respect_to=union,
                is_terminal=True,
            )
    assert False

