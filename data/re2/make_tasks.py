"""
re2: make_tasks.py | Author : Catherine Wong.

Loading tasks and language for the regex domain from Andreas et. al 2017.
"""
import os

from src.task_loaders import *
from data.re2.grammar import *
import dreamcoder.domains.re2.makeRe2Tasks as re2_legacy
from dreamcoder.domains.re2.re2Primitives import *
from dreamcoder.program import Program

DOMAIN_NAME = "re2"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/{DOMAIN_NAME}")
DEFAULT_DATASET = f"{DOMAIN_NAME}_500"
DEFAULT_TASKS_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS)
DEFAULT_LANGUAGE_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, LANGUAGE)


@TaskLoaderRegistry.register
class Re2Loader(TaskDataLoader):
    name = DOMAIN_NAME

    def _get_language_from_task_name(self, task_name):
        return " ".join(task_name.split("_")[3:])

    def load_tasks(self):
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, self.name)

        train_tasks, test_tasks = re2_legacy.loadRe2Dataset(
            task_dataset=DEFAULT_DATASET,
            task_dataset_dir=DEFAULT_TASKS_DIRECTORY,
            type_request=str(tfullstr),
        )

        tasks = {TRAIN: train_tasks, TEST: test_tasks}

        for split in tasks.keys():
            for t in tasks[split]:
                language = self._get_language_from_task_name(t.name)
                t.supervisedSolution = synthetic_language_to_re2_program(language)
                t.groundTruthProgram = t.supervisedSolution
        return tasks


@TaskLanguageLoaderRegistry.register
class Re2HumanLanguageLoader(TaskDataLoader):
    name = "re2_human"

    def load_task_language(self):
        dataset_path = os.path.join(DEFAULT_LANGUAGE_DIRECTORY, DEFAULT_DATASET, HUMAN)
        return self.load_task_language_for_directory(dataset_path, splits=[TRAIN, TEST])


@TaskLanguageLoaderRegistry.register
class Re2SyntheticLanguageLoader(TaskDataLoader):
    name = "re2_synthetic"

    def load_task_language(self):
        dataset_path = os.path.join(
            DEFAULT_LANGUAGE_DIRECTORY, DEFAULT_DATASET, SYNTHETIC
        )
        return self.load_task_language_for_directory(dataset_path, splits=[TRAIN, TEST])

