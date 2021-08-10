"""
compositional_graphics: test_grammar.py | Author : Catherine Wong.
"""

from src.experiment_iterator import ModelLoaderRegistries, GRAMMAR

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

import data.compositional_graphics.grammar as to_test


def test_logo_grammar_load_model():
    grammar_loader = GrammarRegistry[to_test.LogoGrammarLoader.name]
    grammar = grammar_loader.load_model()
    for primitive in grammar.primitives:
        assert grammar_loader.name in str(primitive) or str(primitive).isnumeric()
