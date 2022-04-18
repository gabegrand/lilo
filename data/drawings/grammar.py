"""
drawings: grammar.py | Author : Catherine Wong.

Utility functions for loading Python DSLs for the drawings1k domain.
"""
from collections import OrderedDict
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program

from src.models.model_loaders import ModelLoaderRegistries, GRAMMAR, ModelLoader
from src.models.laps_grammar import LAPSGrammar

import data.drawings.drawings_primitives as drawing_primitives

GrammarRegistry = ModelLoaderRegistries[GRAMMAR]

DOMAIN_PREFIX = "drawings"


class DrawingGrammar(LAPSGrammar):
    @staticmethod
    def new_uniform():
        primitives = (
            drawing_primitives.constants
            + drawing_primitives.math_operations
            + drawing_primitives.transformations
            + drawing_primitives.objects
        )
        grammar = DrawingGrammar.uniform(primitives)
        grammar.function_prefix = f"{DOMAIN_PREFIX}_"
        grammar.__class__ = DrawingGrammar
        return grammar

    @staticmethod
    def render_program(program):
        if type(program) == str:
            program = Program.parse(program)
        return drawing_primitives.render_parsed_program(
            program, allow_partial_rendering=True
        )


@GrammarRegistry.register
class DrawingsGrammarLoader(ModelLoader):
    """Loads the Drawings1K grammar.
    Original source: drawingtasks/gadgets_primitives.py
    """

    name = DOMAIN_PREFIX  # Special handler for OCaml enumeration.

    def load_model(self, experiment_state):
        return DrawingGrammar.new_uniform()
