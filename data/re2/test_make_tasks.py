"""
re2: test_make_tasks | Author : Catherine Wong.
"""

from src.task_loaders import *
import data.re2.make_tasks as to_test


def test_load_re2_graphics_tasks():
    task_loader = TaskLoaderRegistry[to_test.Re2Loader.name]
    tasks = task_loader.load_tasks()
