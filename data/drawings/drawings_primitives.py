"""
drawings_1k_primitives.py | Author : Catherine Wong.

Graphics primitives for rendering gadgets in the Gadgets1K dataset. Based on simple stroke primitives and spatial relations.

Defines Python semantics for DreamCoder primitives: objects are numpy arrays containing an image; transformations are operations on arrays.

Also defines rendering utilities to convert programs and sets of strokes into single images.

This also contains utilities for generating parseable program strings for certain common higher order objects.

Credit: 
- Builds on the primitives from: https://github.com/CatherineWong/drawingtasks/blob/main/primitives/gadgets_primitives.py
- Builds on primitives designed by Lucas Tian in: https://github.com/ellisk42/ec/blob/draw/dreamcoder/domains/draw/primitives.py
"""
import os
import math
import cairo
from data.drawings.tian_primitives import SYNTHESIS_TASK_CANVAS_WIDTH_HEIGHT, XYLIM
import imageio
import numpy as np
from dreamcoder.utilities import Curried
from dreamcoder.program import *
from dreamcoder.type import baseType, arrow, tmaybe, t0, t1, t2

from data.drawings.tian_primitives import (
    tstroke,
    ttransmat,
    _makeAffine,
    _tform_once,
    some_none,
    _repeat,
    _connect,
    transform,
)

tfloat = baseType("tfloat")

## Mathematical operations.
SCALES = np.arange(0.5, 10, 0.25)  # Scaling constants
DISTS = np.arange(-3.0, 3.25, 0.25)  # Distances
INTEGERS = range(0, 13)  # General scaling constants
numeric_constants = set(list(SCALES) + list(DISTS) + list(INTEGERS))
constants = [
    Primitive(f"{n:g}", tfloat, n, override_globals=True) for n in numeric_constants
]
constants += [Primitive("pi", tfloat, math.pi, override_globals=True)]


def _addition(x):
    return lambda y: x + y


def _subtraction(x):
    return lambda y: x - y


def _multiplication(x):
    return lambda y: x * y


def _division(x):
    return lambda y: x / y


def _pow(x):
    return lambda y: x ** y


def _max(x):
    return lambda y: max(x, y)


def _min(x):
    return lambda y: min(x, y)


math_operations = [
    Primitive("-", arrow(tfloat, tfloat, tfloat), _subtraction, override_globals=True),
    Primitive("+", arrow(tfloat, tfloat, tfloat), _addition, override_globals=True),
    Primitive(
        "*", arrow(tfloat, tfloat, tfloat), _multiplication, override_globals=True
    ),
    Primitive("/", arrow(tfloat, tfloat, tfloat), _division, override_globals=True),
    Primitive("pow", arrow(tfloat, tfloat, tfloat), _pow, override_globals=True),
    Primitive("sin", arrow(tfloat, tfloat), math.sin, override_globals=True),
    Primitive("cos", arrow(tfloat, tfloat), math.cos, override_globals=True),
    Primitive("tan", arrow(tfloat, tfloat), math.tan, override_globals=True),
    Primitive("max", arrow(tfloat, tfloat, tfloat), _max, override_globals=True),
    Primitive("min", arrow(tfloat, tfloat, tfloat), _min, override_globals=True),
]

### Basic transform.
# We use a weaker typing than the original in object_primitives.


def _makeAffineSimple(s=1.0, theta=0.0, x=0.0, y=0.0):
    return _makeAffine(s, theta, x, y)


transformations = [
    Primitive(
        "M",  # Makes a transformation matrix
        arrow(
            tfloat,  # Scale
            tfloat,  # Angle
            tfloat,  # Translation X
            tfloat,  # Translation Y
            ttransmat,
        ),
        Curried(_makeAffineSimple),
        alternate_names=["transform_matrix",],
    ),
    Primitive(
        "T",
        arrow(tstroke, ttransmat, tstroke),
        Curried(_tform_once),
        alternate_names=["transform"],
    ),  # Transform: applies a transformation to a stroke array
    Primitive(
        "C",
        arrow(tstroke, tstroke, tstroke),
        Curried(_connect),
        alternate_names=["connect_strokes"],
    ),  # Connects two strokes into a single new primitive
    Primitive(
        "repeat",
        arrow(tstroke, tfloat, ttransmat, tstroke),
        Curried(_repeat),
        override_globals=True,
        alternate_names=["repeat_transform"],
    ),  # Repeats a transformation n times against a base primitive.
]


def connect_strokes(stroke_strings):
    # Utility function to connect several strings into a single stroke. This could be replaced later with a continuation.
    if len(stroke_strings) == 1:
        return stroke_strings[0]

    connected = f"(C {stroke_strings[0]} {stroke_strings[1]})"
    for i in range(2, len(stroke_strings)):
        connected = f"(C {connected} {stroke_strings[i]})"
    return connected


### Basic graphics objects
# Note that any basic graphics object is a list of pixel arrays.
_line = [np.array([(0.0, 0.0), (1.0, 0.0)])]
_circle = [
    np.array(
        [
            (0.5 * math.cos(theta), 0.5 * math.sin(theta))
            for theta in np.linspace(0.0, 2.0 * math.pi, num=30)
        ]
    )
]
_rectangle = [
    np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)])
]


def __scaled_rectangle(width, height):
    strokes = transform(_line, s=width, x=-(width * 0.5), y=height * 0.5) + transform(
        _line, s=width, x=-(width * 0.5), y=-(height * 0.5)
    )
    vertical_line = transform(_line, theta=math.pi / 2)
    strokes += transform(vertical_line, s=height, x=(width * 0.5), y=-(height * 0.5))
    strokes += transform(vertical_line, s=height, x=-(width * 0.5), y=-(height * 0.5))

    return strokes


def _scaled_rectangle(w):
    return lambda h: __scaled_rectangle(w, h)


_emptystroke = []
objects = [
    Primitive("empt", tstroke, _emptystroke, alternate_names=["empty_stroke"],),
    Primitive("l", tstroke, _line, alternate_names=["line"],),
    Primitive("c", tstroke, _circle, alternate_names=["circle"],),
    Primitive("r", tstroke, _rectangle, alternate_names=["square"],),
    Primitive(
        "r_s",
        arrow(tfloat, tfloat, tstroke),
        _scaled_rectangle,
        alternate_names=["rectangle"],
    ),
]

## Higher order utility functions for generating program strings simultaneously with stroke primitives.


def peval(program_string):
    try:
        return float(program_string)
    except:
        p = Program.parse(program_string)
        output = p.evaluate([])
        return output


def get_simplified(program_string):
    try:
        return f"{float(program_string):g}"
    except:
        p = Program.parse(program_string)
        output = p.evaluate([])
        output = f"{output:g}"
        try:
            p = Program.parse(output)
            _ = p.evaluate([])
            return str(output)
        except:
            return program_string


def M_string(s="1", theta="0", x="0", y="0", simplify=False):
    affine_matrix = _makeAffineSimple(peval(s), peval(theta), peval(x), peval(y))
    if simplify:
        m_string = f"(M {get_simplified(s)} {get_simplified(theta)} {get_simplified(x)} {get_simplified(y)})"
    else:
        m_string = f"(M {s} {theta} {x} {y})"
    return affine_matrix, m_string


def T_string(p, p_string, s="1", theta="0", x="0", y="0", simplify=False):
    """Transform Python utility wrapper that applies an affine transformation matrix directly to a primitive, while also generating a string that can be applied to a downstream stroke. Python-usable API that mirrors the functional semantics"""
    tmat, m_string = M_string(s, theta, x, y, simplify=simplify)  # get affine matrix.
    p = _tform_once(p, tmat)
    t_string = f"(T {p_string} {m_string})"
    return p, t_string


def scaled_rectangle_string(w, h, simplify=False):
    if simplify:
        w, h = get_simplified(w), get_simplified(h)
    scaled_rectangle_string = f"(r_s {w} {h})"
    return peval(scaled_rectangle_string), scaled_rectangle_string


def polygon_string(n, simplify=False):
    if simplify:
        n = get_simplified(str(n))
    y = f"(/ 0.5 (tan (/ pi {n})))"
    theta = f"(/ (* 2 pi) {n})"

    # Base line that forms the side.
    _, base_line = T_string(_line, "l", x="-0.5", y=y, simplify=simplify)

    # Rotation
    _, rotation = M_string(theta=theta, simplify=simplify)

    polygon_string = f"(repeat {base_line} {n} {rotation})"
    return peval(polygon_string), polygon_string


def nested_scaling_string(shape_string, n, scale_factor):
    # Scale factor
    _, scale = M_string(s=scale_factor)
    nested_scaling_string = f"(repeat {shape_string} {n} {scale})"

    return peval(nested_scaling_string), nested_scaling_string


def rotation_string(
    p, p_string, n, displacement="0.5", decorator_start_angle="(/ pi 4)", simplify=False
):

    if simplify:
        n = get_simplified(n)
        displacement = get_simplified(displacement)
    y = f"(* {displacement} (sin {decorator_start_angle}))"
    x = f"(* {displacement} (cos {decorator_start_angle}))"
    theta = f"(/ (* 2 pi) {n})"

    # Base line that forms the side.
    _, base_object = T_string(p, p_string, x=x, y=y, simplify=simplify)

    # Rotation
    _, rotation = M_string(theta=theta, simplify=simplify)

    rotated_object_string = f"(repeat {base_object} {n} {rotation})"
    return peval(rotated_object_string), rotated_object_string


c_string = (
    _circle,
    "c",
)  # Circle
r_string = (_rectangle, "r")  # Rectangle
cc_string = T_string(c_string[0], c_string[-1], s="2")  # Double scaled circle
hexagon_string = polygon_string(6)
octagon_string = polygon_string(8)
l_string = (_line, "l")
short_l_string = T_string(
    l_string[0], l_string[-1], x="(- 0 0.5)"
)  # Short horizontal line


### Shape wrapper class. Contains sub-fields for the stroke array, base DSL program, synthetic language, and synthetic abstractions associated with the shape.
LANG_A = "a"
LANG_TINY, LANG_SMALL, LANG_MEDIUM, LANG_LARGE, LANG_VERY_LARGE = (
    "tiny",
    "small",
    "medium",
    "large",
    "very large",
)
LANG_SIZES = [LANG_TINY, LANG_SMALL, LANG_MEDIUM, LANG_LARGE, LANG_VERY_LARGE]
LANG_CIRCLE, LANG_RECTANGLE, LANG_LINE, LANG_SQUARE = (
    "circle",
    "rectangle",
    "line",
    "square",
)
LANG_CENTER_OF_THE_IMAGE = "center of the image"
LANG_GON_NAMES = {
    3: "triangle",
    4: "square",
    5: "pentagon",
    6: "hexagon",
    7: "septagon",
    8: "octagon",
    9: "nonagon",
}


# Utilities for rendering.
def render_stroke_arrays_to_canvas(
    stroke_arrays,
    stroke_width_height=8 * XYLIM,
    canvas_width_height=SYNTHESIS_TASK_CANVAS_WIDTH_HEIGHT,
):
    """See original source: prog2pxl https://github.com/ellisk42/ec/blob/draw/dreamcoder/domains/draw/primitives.py"""
    scale = canvas_width_height / stroke_width_height

    canvas_array = np.zeros((canvas_width_height, canvas_width_height), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(
        canvas_array, cairo.Format.A8, canvas_width_height - 2, canvas_width_height - 2
    )
    context = cairo.Context(surface)
    context.set_source_rgb(512, 512, 512)

    for stroke_array in stroke_arrays:
        renderable_stroke = np.copy(stroke_array)
        renderable_stroke += stroke_width_height / 2  # Centering
        renderable_stroke *= scale
        for pixel in renderable_stroke:
            context.line_to(pixel[0], pixel[1])
        context.stroke()
    return np.flip(canvas_array, 0) / (canvas_width_height * 2)


def render_parsed_program(
    program,
    stroke_width_height=8 * XYLIM,
    canvas_width_height=SYNTHESIS_TASK_CANVAS_WIDTH_HEIGHT,
):
    if type(program) == str:
        program = Program.parse(program)
    if not hasattr(program, "rendering"):
        program.rendering = render_stroke_arrays_to_canvas(
            program.evaluate([]), stroke_width_height, canvas_width_height
        )
    return program.rendering


def export_rendered_program(rendered_array, export_id, export_dir):
    filename = os.path.join(export_dir, f"{export_id}.png")
    b_w_array = np.array((rendered_array > 0)).astype(int)  # Make black and white.
    inverted_array = 1 - b_w_array  # Invert B/W image for aesthetics.
    imageio.imwrite(filename, inverted_array)
    return filename
