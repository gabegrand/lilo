"""
clevr: grammar.py | Author: Catherine Wong.
Utility functions for loading in the DSLs for the CLEVR domain.
"""
from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.clevr.clevrPrimitives as clevrPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class ClevrGrammarLoader(ModelLoader):
    """
    Loads the CLEVR grammar.
    Original source: dreamcoder/domains/clevrPrimitives
    """

    name = "clevr"  # Special handler for OCaml enumeration.
    # Original ICML paper: uses list primitives and no filter.
    def load_model(self, experiment_state):
        CLEVR_PRIMITIVES = clevrPrimitives.load_clevr_primitives(
            ["clevr_bootstrap", "clevr_map_transform"]
        )
        grammar = LAPSGrammar.uniform(CLEVR_PRIMITIVES)
        grammar.function_prefix = "clevr"
        return grammar
