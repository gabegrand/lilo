"""
re2: grammar.py | Author : Catherine Wong.
Utility functions for loading in the DSLs for the regex domain.
"""
from collections import OrderedDict
from nis import match

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.re2.re2Primitives as re2Primitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class Re2GrammarLoader(ModelLoader):
    """Loads the Re2 grammar.
    Original source: dreamcoder/domains/re2/re2Primitives.
    Semantics are only implemented in OCaml.
    """

    name = "re2"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        # Give it re2_chars_None
        # Give it re2_bootstrap_v1_primitives.
        # Don't give vowel consonant.

        logo_primitives = list(
            OrderedDict((x, True) for x in logoPrimitives.primitives).keys()
        )
        grammar = LAPSGrammar.uniform(
            logo_primitives, continuationType=logoPrimitives.turtle
        )
        return grammar


# Utility functions to parse the synthetic language back into regexes.
STARTS_WITH = ("if the word starts with",)
ENDS_WITH = "if the word ends with"
ANYWHERE = "if there is"
VOWEL, CONSONANT, ANY_LETTER = "vowel", "consonant", "any letter"
ANY_REGEX = "."
MATCH_TOKENS = [VOWEL, CONSONANT, ANY_REGEX] + [
    f"{chr(i)}" for i in range(ord("a"), ord("z"))
]
VOWEL_MATCH_REGEX = "(_ror (_ror (_ror (_ror _a _e) _i) _o) _u)"
CONSONANT_MATCH_REGEX = (
    "(_rnot (_rconcat (_rconcat (_rconcat (_rconcat _a _e) _i) _o) _u))"
)
REPLACE_MATCH = "replace that with"
DOUBLE_MATCH = "double that"
ADD = "add"
POSTPEND_MATCH = "after that"
PREPEND_MATCH = "before that"
REMOVE_MATCH = "remove that"


def synthetic_language_to_re2_program(language):
    match_type, match_tokens = get_match_type(language)
    replace_type, replace_tokens = get_replacement_type(language)

    # Build the match regex.
    match_regex = build_match_regex(match_tokens)

    # Build the replacement.

    # Apply the


def get_match_type(language):
    """:ret: match_type, match regex string."""
    language = language.replace(ANY_LETTER, ANY_REGEX)
    match_tokens = []
    for match_type in [STARTS_WITH, ENDS_WITH, ANYWHERE]:
        if language.startswith(match_type):
            for token in language.split(match_type)[1:][0].split():
                if token not in MATCH_TOKENS:
                    return match_type, match_tokens
                else:
                    match_tokens.append(token)

    assert False


def get_replacement_type(language, match_tokens):
    # Replace match.
    language = language.replace(ANY_LETTER, ANY_REGEX)
    match_tokens = []
    if REPLACE_MATCH in language:
        match_tokens = language.split(REPLACE_MATCH)[1:][0].split()
        return REPLACE_MATCH, match_tokens

    # Prepend X to match.

    # Postpend X to match.

    # Remove match.

    # Double match.


def token_to_regex(token):
    if token == VOWEL:
        return VOWEL_MATCH_REGEX
    if token == CONSONANT:
        return CONSONANT_MATCH_REGEX
    if token == ANY_LETTER:
        return ANY_LETTER
    return "_" + token


def build_match_regex(match_tokens):
    match_tokens = [token_to_regex(t) for t in match_tokens]
    if len(match_tokens) == 1:
        return f"{match_tokens[0]}"
    elif len(match_tokens) == 2:
        return f"(_rconcat {match_tokens[0]} {match_tokens[1]})"
    else:
        assert False


def build_replace_regex(replace_type, match_tokens, replace_tokens):
    match_regex = build_match_regex(match_tokens)
    replace_regex = build_match_regex(replace_tokens)
    if replace_type == REPLACE_MATCH:
        return f"(lambda  (if (_rmatch {match_regex} $0) {replace_regex} $0)   )"
    elif replace_type == PREPEND_MATCH:
        return f"(lambda  (if (_rmatch {match_regex} $0) (_rconcat {replace_regex} $0) $0)   )"
    elif replace_type == POSTPEND_MATCH:
        return f"(lambda  (if (_rmatch {match_regex} $0) (_rconcat $0 {replace_regex}) $0)   )"
    elif replace_type == REMOVE_MATCH:
        return f"(lambda  (if (_rmatch {match_regex} $0) _emptystr $0) )"
    else:
        assert False


def build_full_regex(match_type, replace_regex, match_tokens):
    match_regex = build_match_regex(match_tokens)
    if match_type == ANYWHERE:
        return (
            f"(lambda (_rflatten (map  {replace_regex}  (_rsplit {match_regex} $0) ) ))"
        )
    elif match_type == STARTS_WITH:
        return f"(lambda ((lambda (_rflatten (cons ({replace_regex} (car $0)) (cdr $0)) )) (_rsplit {match_regex} $0) ))"
    elif match_type == ENDS_WITH:
        return f"(lambda ((lambda (_rflatten (_rappend ({replace_regex} (_rtail $0)) (_rrevcdr $0)) )) (_rsplit {match_regex} $0) ))"
    else:
        assert False
