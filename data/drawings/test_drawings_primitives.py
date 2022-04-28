"""
test_drawings_primitives.py | Author : Catherine Wong.
"""
import os
import data.drawings.drawings_primitives as to_test

from dreamcoder.program import Program

OUTPUT = "data/drawings/test_renders"


def _test_parse_render_save_shape_programs(shapes, tmpdir, split="train"):
    program_strings = [shape.base_program for shape in shapes]
    _test_parse_render_save_programs(program_strings, tmpdir, split)


def _test_parse_render_save_programs(program_strings, tmpdir, split="train"):
    export_dir = tmpdir
    for program_id, program_string in enumerate(program_strings):
        color = (255, 0, 0)
        try:
            # Can it parse the program?
            p = Program.parse(program_string)
            # Can it render the program?
            rendered = to_test.render_parsed_program(
                p, stroke_width_height=8 * to_test.XYLIM, color=color
            )
            assert rendered.shape == (
                to_test.SYNTHESIS_TASK_CANVAS_WIDTH_HEIGHT,
                to_test.SYNTHESIS_TASK_CANVAS_WIDTH_HEIGHT,
                3,
            )
            # Can it save the program?
            saved_file = to_test.export_rendered_program(
                rendered, f"{split}_{program_id}", export_dir=export_dir, color=color
            )
            assert os.path.exists(saved_file)
        except Exception as e:
            print(f"Failed to evaluate: {program_string}, {str(e)}")
            assert False


def test_program_rendering():
    test_programs = [
        "(C (T (T c (M 2 0 0 0)) (M 2 0 0 0)) (T (T c (M 2 0 0 0)) (M 1 0 0 0)))",
        "(C (T (T c (M 2 0 0 0)) (M 2 0 0 0)) (T (T c (M 2 0 0 0)) (M 0.5 0 0 0)))",
        "(C (C (T (repeat (T l (M 1 0 -0.5 (/ 0.5 (tan (/ pi 6))))) 6 (M 1 (/ (* 2 pi) 6) 0 0)) (M 2 0 0 0)) (T (repeat (T l (M 1 0 -0.5 (/ 0.5 (tan (/ pi 6))))) 6 (M 1 (/ (* 2 pi) 6) 0 0)) (M 2.25 0 0 0))) (T (repeat (T l (M 1 0 -0.5 (/ 0.5 (tan (/ pi 6))))) 6 (M 1 (/ (* 2 pi) 6) 0 0)) (M 0.5 0 0 0)))",
        "(C (C (T (repeat (T l (M 1 0 -0.5 (/ 0.5 (tan (/ pi 8))))) 8 (M 1 (/ (* 2 pi) 8) 0 0)) (M 4 0 0 0)) (T (T c (M 2 0 0 0)) (M 1 0 0 0))) (repeat (T (T c (M 0.5 0 0 0)) (M 1 0 (* (* 1 1.5) (cos (/ pi 4))) (* (* 1 1.5) (sin (/ pi 4))))) 6 (M 1 (/ (* 2 pi) 6) 0 0)))",
        "(C (C (C (C (C (T (T (r_s 2 3.375) (M 1 0 0 (* 3.375 0.5))) (M 1 0 (+ -10 (* 0.5 2)) 0)) (T (T (r_s 16 4.5) (M 1 0 0 (* 4.5 0.5))) (M 1 0 (+ -8 (* 0.5 16)) 0))) (T (T (r_s 2 3.375) (M 1 0 0 (* 3.375 0.5))) (M 1 0 (+ 8 (* 0.5 2)) 0))) (T (r_s 12 1) (M 1 0 (+ 0 0) (+ (+ 4.5 (+ (- 0 0) (* 1 0.5))) 0)))) (T (C (C (C (T (T (T l (M 3 (/ pi 2) 0 (- 0 2))) (M 1 0 0 0)) (M 1 0 0 0)) (T (T (T l (M 1 0 (- 0 0.5) 0)) (M (- (* 1 2) 0) 0 0 0)) (M 1 0 0 (- (* (* 0.5 1) 2) 0)))) (T (T (T l (M 1 0 (- 0 0.5) 0)) (M (- (* 1 2) 1) 0 0 0)) (M 1 0 0 (- (* (* 0.5 1) 2) 1)))) (T (T (T l (M 1 0 (- 0 0.5) 0)) (M (- (* 1 2) 2) 0 0 0)) (M 1 0 0 (- (* (* 0.5 1) 2) 2)))) (M 1 0 (+ 0 0) (+ (+ 5.5 (+ (- 0 0) (* 5 0.5))) 0)))) (T (repeat (repeat (T (C (C (C (T c (M 2.25 0 0 0)) (T c (M 2.75 0 0 0))) (T r (M 0.5 0 0 0))) (repeat (T (T c (M 0.25 0 0 0)) (M 1 0 (* 0.75 (cos (/ pi 4))) (* 0.75 (sin (/ pi 4))))) 4 (M 1 (/ (* 2 pi) 4) 0 0))) (M 1 0 0 0)) 6 (M 1 0 (/ (- (- 8 (* 0.5 3.25)) (+ -8 (* 0.5 3.25))) 5) 0)) 1 (M 1 0 0 0)) (M 1 0 (/ (- (- 8 (* 0.5 3.25)) (+ -8 (* 0.5 3.25))) -2) (/ (- 0 0) -2))))",
    ]

    _test_parse_render_save_programs(program_strings=test_programs, tmpdir=OUTPUT)
