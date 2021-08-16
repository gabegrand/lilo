"""
tower: test_grammar.py | Author : Catherine Wong.
"""

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, GrammarLoader


GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

import data.tower.grammar as to_test


def test_tower_grammar_load_model():
    grammar_loader = GrammarRegistry[to_test.TowerGrammarLoader.name]
    grammar = grammar_loader.load_model()
    assert len(grammar.primitives) > 0
