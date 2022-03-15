"""
compositional_graphics: grammar.py | Author : Catherine Wong.

Utility functions for loading Python DSLs for the compositional graphics domain. This grammar was originally designed and used in the LAPS-ICML 2021 paper and can be found in the dreamcoder/logo domain.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import dreamcoder.domains.logo.logoPrimitives as logoPrimitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class LogoGrammarLoader(ModelLoader):
    """Loads the LOGO grammar.
    Original source: dreamcoder/domains/logo/logoPrimitives.
    Semantics are only implemented in OCaml.
    """

    name = "LOGO"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        logo_primitives = list(
            OrderedDict((x, True) for x in logoPrimitives.primitives).keys()
        )
        grammar = LAPSGrammar.uniform(
            logo_primitives, continuationType=logoPrimitives.turtle
        )
        grammar.function_prefix = "logo"
        return grammar
