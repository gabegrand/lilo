"""
tower: grammar.py | Author : Catherine Wong.

Utility functions for loading Python DSLs for the tower domain. This grammar was originally designed and used in the DreamCoder 2021 paper and can be found in the dreamcoder/tower domain.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, GrammarLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.tower.towerPrimitives as towerPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class TowerGrammarLoader(GrammarLoader):
    """Loads the tower grammar.
    Original source: dreamcoder/domains/tower/towerPrimitives.
    """

    name = "tower"  # Special handler for OCaml enumeration.

    def load_model(self):
        tower_primitives = list(
            OrderedDict((x, True) for x in towerPrimitives.primitives).keys()
        )
        grammar = LAPSGrammar.uniform(
            tower_primitives, continuationType=towerPrimitives.ttower
        )
        return grammar
