"""
structures: test_make_tasks.py | Author : Catherine Wong.
"""

from dreamcoder.task import Task
from src.task_loaders import *
import data.structures.make_tasks as to_test


def test_make_structures_tasks():
    for subdomain in to_test.SUBDOMAINS:
        task_loader = TaskLoaderRegistry[to_test.Structures1KLoader.name]
        dataset_path = os.path.join(
            to_test.DEFAULT_DATA_DIRECTORY, TASKS, task_loader.name
        )
        task_loader.make_structures_tasks(subdomain, dataset_path)


def test_load_tasks():
    task_loader = TaskLoaderRegistry[to_test.Structures1KLoader.name]
    tasks = task_loader.load_tasks()
    assert len(tasks[TRAIN]) == 800
    assert len(tasks[TEST]) == 200


def test_make_structures_language():
    task_loader = TaskLanguageLoaderRegistry[
        to_test.Structures1KHumanLanguageLoader.name
    ]
    dataset_path = os.path.join(
        to_test.DEFAULT_DATA_DIRECTORY,
        LANGUAGE,
        to_test.Structures1KLoader.name,
        HUMAN,
    )
    task_loader.make_structures_language(dataset_path)
