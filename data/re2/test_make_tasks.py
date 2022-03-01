"""
re2: test_make_tasks | Author : Catherine Wong.
"""

from src.task_loaders import *
import data.re2.make_tasks as to_test


def test_load_re2_tasks():
    task_loader = TaskLoaderRegistry[to_test.Re2Loader.name]
    tasks = task_loader.load_tasks()
    for split in tasks:
        print(split, len(tasks[split]))
        for task in tasks[split]:
            assert task.supervisedSolution is not None


def test_load_re2_language_human():
    language_loader = TaskLanguageLoaderRegistry[to_test.Re2HumanLanguageLoader.name]
    language, _ = language_loader.load_task_language()

    task_loader = TaskLoaderRegistry[to_test.Re2Loader.name]
    tasks = task_loader.load_tasks()
    for split in (TRAIN, TEST):
        for task in tasks[split]:
            assert task.name in language[split]


def test_load_re2_language_synthetic():
    language_loader = TaskLanguageLoaderRegistry[
        to_test.Re2SyntheticLanguageLoader.name
    ]
    language, _ = language_loader.load_task_language()

    task_loader = TaskLoaderRegistry[to_test.Re2Loader.name]
    tasks = task_loader.load_tasks()
    for split in (TRAIN, TEST):
        for task in tasks[split]:
            assert task.name in language[split]
