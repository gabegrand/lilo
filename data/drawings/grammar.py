"""
drawings: grammar.py | Author : Catherine Wong.

Utility functions for loading Python DSLs for the drawings1k domain.
"""
from collections import OrderedDict

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import data.drawings.drawings_primitives as drawing_primitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]


@GrammarRegistry.register
class DrawingsGrammarLoader(ModelLoader):
    """Loads the Drawings1K grammar.
    Original source: drawingtasks/gadgets_primitives.py
    """

    name = "drawings"  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        drawing_primitives = (
            drawing_primitives.constants
            + drawing_primitives.math_operations
            + drawing_primitives.transformations
            + drawing_primitives.objects
        )
        grammar = LAPSGrammar.uniform(drawing_primitives,)
        grammar.function_prefix = "drawing_"

        # Add on a rendering function.

        return grammar

