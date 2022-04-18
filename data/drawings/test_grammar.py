"""drawings:test_grammar.py"""
import numpy as np
import data.drawings.grammar as to_test


def test_new_drawing_grammar():
    grammar = to_test.DrawingGrammar.new_uniform()
    assert len(grammar.productions) > 0

    # Test parsing.
    test_program = "(C (C (T (T c (M 2 0 0 0)) (M 4 0 0 0)) (T (T c (M 2 0 0 0)) (M 4.25 0 0 0))) (T (repeat (T l (M 1 0 -0.5 (/ 0.5 (tan (/ pi 6))))) 6 (M 1 (/ (* 2 pi) 6) 0 0)) (M 2 0 0 0)))"
    rendered = grammar.render_program(test_program)
    assert np.sum(rendered) > 0
