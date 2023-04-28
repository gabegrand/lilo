"""
compositional_graphics: make_tasks.py | Author: Catherine Wong.

Utility functions for loading tasks and language for the compositional graphics domain. This domain was originally collected and used in the LAPS-ICML 2021 paper and can be found in the dreamcoder/logo domain.
"""
import os

import dill

from dreamcoder.domains.logo.makeLogoTasks import drawLogo
from dreamcoder.task import Task
from src.task_loaders import *

ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, "data/compositional_graphics")
OCAML_SPECIAL_HANDLER_LOGO = "LOGO"  # Special string flag for OCaml handling.


class CompositionalGraphicsTask(Task):
    """Wrapper around dreamcoder.task.Task that overrides `check()` to use `drawLogo()`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_task(t):
        task = CompositionalGraphicsTask(
            t.name, t.request, t.examples, features=t.features, cache=t.cache
        )
        for attr in dir(t):
            if not hasattr(task, attr):
                task.__setattr__(attr, getattr(t, attr))
        return task

    def check(self, e, timeout=None):
        for _, y in self.examples:
            yh = drawLogo(e, timeout=timeout, resolution=28)
            if yh != y:
                return False
        return True


@TaskLoaderRegistry.register
class CompositionalGraphics200Loader(TaskDataLoader):
    name = "compositional_graphics_200"

    def load_tasks(self):
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, self.name)
        tasks = {TRAIN: [], TEST: []}
        for split in tasks.keys():
            split_path = os.path.join(dataset_path, split)
            task_files = sorted(
                [
                    os.path.join(split_path, t)
                    for t in os.listdir(split_path)
                    if ".p" in t
                ],
                key=lambda p: os.path.basename(p),
            )
            for task_file in task_files:
                with open(task_file, "rb") as f:
                    task_dill = dill.load(f)
                    t = CompositionalGraphicsTask.from_task(task_dill)

                    t.nearest_name = None
                    # Add the task serializer.
                    t.ocaml_serializer = None
                    # Add supervision.
                    t.supervisedSolution = t.groundTruthProgram

                    tasks[split].append(t)
        return tasks


@TaskLanguageLoaderRegistry.register
class CompositionalGraphics200HumanLanguageLoader(TaskDataLoader):
    name = "compositional_graphics_200_human"

    def load_task_language(self):
        task_dataset = CompositionalGraphics200Loader.name
        dataset_path = os.path.join(
            DEFAULT_DATA_DIRECTORY, LANGUAGE, task_dataset, HUMAN
        )
        return self.load_task_language_for_directory(dataset_path, splits=[TRAIN, TEST])


@TaskLanguageLoaderRegistry.register
class CompositionalGraphics200SyntheticLanguageLoader(TaskDataLoader):
    name = "compositional_graphics_200_synthetic"

    def load_task_language(self):
        task_dataset = CompositionalGraphics200Loader.name
        dataset_path = os.path.join(
            DEFAULT_DATA_DIRECTORY, LANGUAGE, task_dataset, SYNTHETIC
        )
        return self.load_task_language_for_directory(dataset_path, splits=[TRAIN, TEST])
