"""
drawings: test_make_tasks.py | Author : Catherine Wong
"""
from src.task_loaders import *

import data.drawings.make_tasks as to_test


def test_load_tasks_for_domain():
    task_loader = TaskLoaderRegistry[to_test.DrawingsLoader.name]
    tasks = task_loader.load_tasks_for_domain("nuts_bolts")
    for split in [TRAIN, TEST]:
        assert len(tasks[split]) > 0


def test_load_tasks_all():
    task_loader = TaskLoaderRegistry[to_test.DrawingsLoader.name]
    tasks = task_loader.load_tasks()
    for split in [TRAIN, TEST]:
        assert len(tasks[split]) > 0

