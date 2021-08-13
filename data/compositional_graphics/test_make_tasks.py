"""
compositional_graphics: test_make_tasks.py | Author : Catherine Wong.
"""

from src.task_loaders import *
from dreamcoder.domains.logo.logoPrimitives import turtle
from dreamcoder.type import arrow

import data.compositional_graphics.make_tasks as to_test


def test_load_compositional_graphics_200_tasks():
    task_loader = TaskLoaderRegistry[to_test.CompositionalGraphics200Loader.name]
    tasks = task_loader.load_tasks()
    assert len(tasks[TRAIN]) == 200
    assert len(tasks[TEST]) == 111
    for split in (TRAIN, TEST):
        for task in tasks[split]:
            assert task.request == arrow(turtle, turtle)


def test_load_compositional_graphics_200_human_language():
    language_loader = TaskLanguageLoaderRegistry[
        to_test.CompositionalGraphics200HumanLanguageLoader.name
    ]
    language, _ = language_loader.load_task_language()

    task_loader = TaskLoaderRegistry[to_test.CompositionalGraphics200Loader.name]
    tasks = task_loader.load_tasks()
    for split in (TRAIN, TEST):
        for task in tasks[split]:
            assert task.name in language[split]


def test_load_compositional_graphics_200_synthetic_language():
    language_loader = TaskLanguageLoaderRegistry[
        to_test.CompositionalGraphics200SyntheticLanguageLoader.name
    ]
    language, _ = language_loader.load_task_language()

    task_loader = TaskLoaderRegistry[to_test.CompositionalGraphics200Loader.name]
    tasks = task_loader.load_tasks()
    for split in (TRAIN, TEST):
        for task in tasks[split]:
            assert task.name in language[split]
