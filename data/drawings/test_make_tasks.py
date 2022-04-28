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


def test_domain_load_tasks():
    for domain_name in to_test.TASK_DOMAINS:
        task_loader = TaskLoaderRegistry[
            to_test.DrawingsLoader.name + "_" + domain_name
        ]
        tasks = task_loader.load_tasks()
        assert len(tasks[TRAIN]) == 200


def test_build_language_vocabulary_jsons():
    language_loader = to_test.DrawingsHumanLanguageLoader()
    domain_dataset_path = language_loader.build_language_vocabulary_jsons("nuts_bolts")
    assert os.path.exists(domain_dataset_path)


def test_load_task_language_domain():
    language_loader = to_test.DrawingsHumanLanguageLoader()
    for domain in to_test.TASK_DOMAINS:
        language, vocab = language_loader.load_task_language_domain(domain)
        assert len(language[TRAIN]) == 200
        assert len(language[TEST]) == 50
        assert len(vocab[TRAIN]) > 0
        assert len(vocab[TEST]) > 0


def test_load_task_language_all():
    language_loader = to_test.DrawingsHumanLanguageLoader()
    language, vocab = language_loader.load_task_language()
    assert len(language[TRAIN]) == 800
    assert len(language[TEST]) == 200
    assert len(vocab[TRAIN]) > 0
    assert len(vocab[TEST]) > 0


def test_domain_language_load_tasks():
    for domain_name in to_test.TASK_DOMAINS:

        task_loader = TaskLanguageLoaderRegistry[
            to_test.DrawingsHumanLanguageLoader.name + "_" + domain_name
        ]
        language, vocab = task_loader.load_task_language()
        assert len(language[TRAIN]) == 200
