"""
compositional_graphics: test_grammar.py | Author : Catherine Wong.
"""

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader


GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

import data.compositional_graphics.grammar as to_test


def test_logo_grammar_load_model():
    grammar_loader = GrammarRegistry[to_test.LogoGrammarLoader.name]
    grammar = grammar_loader.load_model()
    for primitive in grammar.primitives:
        assert (
            grammar_loader.name.lower() in str(primitive).lower()
            or str(primitive).isnumeric()
        )
