"""object_primitives.py | Author : Catherine Wong.
Graphics primitives based on objects and spatial relations. Places predefined objects on a grid.

Defines Python semantics for DreamCoder primitives: objects are numpy arrays containing an image; transformations are operations on arrays.

Also defines rendering utilities to convert programs and sets of strokes into single images.

Credit: builds on primitives designed by Lucas Tian in: https://github.com/ellisk42/ec/blob/draw/dreamcoder/domains/draw/primitives.py
"""
import os
import math
import cairo
import imageio
import numpy as np
from dreamcoder.utilities import Curried
from dreamcoder.program import Program, Primitive
from dreamcoder.type import baseType, arrow, tmaybe, t0, t1, t2

### Base types
tstroke = baseType("tstroke")
tangle = baseType("tangle")
tscale = baseType("tscale")
tdist = baseType("tdist")
ttrorder = baseType("ttorder")
ttransmat = baseType("ttransmat")
trep = baseType("trep")

### Constant values
XYLIM = 3.5  # i.e., -3 to 3
SYNTHESIS_TASK_CANVAS_WIDTH_HEIGHT = 512

SCALES = [0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
NPOLY = range(3, 7)  # range of regular polyogns allowed.
DISTS = (
    [-2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    + [-1.75, -0.65, 0.45, 1.55, 1.1]
    + [0.5 / math.tan(math.pi / n) for n in range(3, 7)]
)
THETAS = (
    [j * (2 * math.pi / 8) for j in range(8)] + [-2 * math.pi / 6] + [-2 * math.pi / 12]
)
ORDERS = ["trs", "tsr", "rts", "rst", "srt", "str"]

scales = [Primitive("scale{}".format(i), tscale, j) for i, j in enumerate(SCALES)]
distances = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(DISTS)]
angles = [Primitive("angle{}".format(i), tangle, j) for i, j in enumerate(THETAS)]
orders = [Primitive(j, ttrorder, j) for j in ORDERS]
repetitions = [
    Primitive("rep{}".format(i), trep, j + 1) for i, j in enumerate(range(7))
]
constants = scales + distances + angles + orders + repetitions

### Some and None
def _return_argument(argument):
    return argument


some_none = [
    Primitive("None", tmaybe(t0), None),
    Primitive("Some", arrow(t0, tmaybe(t0)), _return_argument),
]

### Basic graphics objects
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

_emptystroke = []
objects = [
    Primitive("emptystroke", tstroke, _emptystroke),
    Primitive("line", tstroke, _line),
    Primitive("circle", tstroke, _circle),
    Primitive("rectangle", tstroke, _rectangle),
]

# Higher order utility functions. These are composed of lines.
def polygon(n=3):
    # Regular polygons
    y = 0.5 / math.tan(math.pi / n)
    return _repeat(transform(_line, x=-0.5, y=y), n, _makeAffine(theta=2 * math.pi / n))


def rectangle(width, height):
    strokes = transform(_line, s=width, x=-(width * 0.5), y=height * 0.5) + transform(
        _line, s=width, x=-(width * 0.5), y=-(height * 0.5)
    )
    vertical_line = transform(_line, theta=math.pi / 2)
    strokes += transform(vertical_line, s=height, x=(width * 0.5), y=-(height * 0.5))
    strokes += transform(vertical_line, s=height, x=-(width * 0.5), y=-(height * 0.5))

    return strokes


### Transformations over objects. Original source from https://github.com/ellisk42/ec/blob/draw/dreamcoder/domains/draw/primitives.py
def set_default_if_none(arg, default):
    return arg if arg is not None else default


def _makeAffine(s=1.0, theta=0.0, x=0.0, y=0.0, order=ORDERS[0]):
    """Makes an affine transformation matrix for any linear combination of translation, rotation, scaling.
    :order: one of the 6 ways you can permutate the three transformation primitives.
    Passed as a string (e.g. "trs" means scale, then rotate, then tranlate.)
    Input and output types guarantees a primitive will only be transformed once.
    """
    s = set_default_if_none(s, 1.0)
    theta = set_default_if_none(theta, 0.0)
    x = set_default_if_none(x, 0.0)
    y = set_default_if_none(y, 0.0)
    order = set_default_if_none(order, ORDERS[0])

    def _rotation(theta):
        transformation_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return transformation_matrix

    def _scale(s):  # 2D uniform scaling in x and y
        transformation_matrix = np.array(
            [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]]
        )
        return transformation_matrix

    def _translation(x, y):
        transformation_matrix = np.array(
            [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]]
        )
        return transformation_matrix

    char_to_function = {"t": _translation(x, y), "r": _rotation(theta), "s": _scale(s)}

    def _parse_order_string(order):
        transformation_fns = [char_to_function[char] for char in order]
        return transformation_fns[0] @ transformation_fns[1] @ transformation_fns[2]

    return _parse_order_string(order)


def _tform_iterative(p, transformation_matrix, i=1):
    """Applies a transformation matrix i times to the object p.
    :p: can be a list or numpy array.
    """
    if isinstance(p, list):
        return [_tform_iterative(x, transformation_matrix, i) for x in p]

    p = np.concatenate((p, np.ones((p.shape[0], 1))), axis=1)
    for _ in range(i):
        p = (
            transformation_matrix @ p.transpose()
        ).transpose()  # apply affine transfomatiton.
    p = np.delete(p, 2, axis=1)  # --- remove third dimension
    return p


def _tform_once(p, transformation_matrix):
    return _tform_iterative(p, transformation_matrix, i=1)


def transform(p, s=1.0, theta=0.0, x=0.0, y=0.0, order=ORDERS[0]):
    """Transform Python utility wrapper that applies an affine transformation matrix directly to a primitive. Python-usable API that mirrors the functional semantics"""
    T = _makeAffine(s, theta, x, y, order)  # get affine matrix.
    p = _tform_once(p, T)
    return p


transformations = [
    Primitive(
        "transmat",
        arrow(
            tmaybe(tscale),
            tmaybe(tangle),
            tmaybe(tdist),
            tmaybe(tdist),
            tmaybe(ttrorder),
            ttransmat,
        ),
        Curried(_makeAffine),
    ),
    Primitive("transform", arrow(tstroke, ttransmat, tstroke), Curried(_tform_once)),
]

## Complex relational primitives
def _reflect(p, theta=math.pi / 2):
    """Applies a reflection to object p over the line through the origin. Rotates p by -theta, then reflects it over the y axis and unrotates by +theta. Y-axis is pi/2."""

    th = theta - math.pi / 2
    p = transform(p, theta=-th)
    T = np.array([[-1.0, 0.0], [0.0, 1.0]])
    p = [np.matmul(T, pp.transpose()).transpose() for pp in p]
    p = transform(p, theta=th)
    return p


def _repeat(p, n, transformation_matrix):
    """
    Takes a base primitive p and returns a list of n primitives, each which transforms the n-1 primitive in the list by the transformation_matrix.
    """
    p_out = []
    for i in range(int(n)):
        if i > 0:
            p = _tform_once(p, transformation_matrix)  # apply transformation
        pthis = [np.copy(pp) for pp in p]  # copy current state, and append
        p_out.extend(pthis)
    return p_out


def _connect(p1, p2):
    """Connects two primitives into a single new primitive."""
    return p1 + p2


relations = [
    Primitive("reflect", arrow(tstroke, tangle, tstroke), Curried(_reflect)),
    Primitive("connect", arrow(tstroke, tstroke, tstroke), Curried(_connect)),
    Primitive("repeat", arrow(tstroke, trep, ttransmat, tstroke), Curried(_repeat)),
]


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
