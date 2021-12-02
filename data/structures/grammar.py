"""
structures: grammar.py | Author : Catherine Wong.

Utility functions for loading Python DSLs for the structures domain. This grammar was originally designed and used in the LAPS-ICML 2021 paper and can be found in the dreamcoder/tower domain.

Unlike the tower domain, we use 2x1 blocks and a larger set of integer primitives.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.tower.towerPrimitives as towerPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class SupervisedTowerGrammarLoader(ModelLoader):
    """Loads the tower grammar.
    Original source: dreamcoder/domains/tower/towerPrimitives.
    Semantics are only implemented in OCaml.
    """

    name = "supervisedTower"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        grammar = LAPSGrammar.uniform(
            towerPrimitives.primitives, continuationType=towerPrimitives.ttower
        )
        return grammar
