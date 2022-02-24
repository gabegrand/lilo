"""
re2: make_tasks.py | Author : Catherine Wong.

Loading tasks and language for the regex domain from Andreas et. al 2017.
"""
import os

import dill

from src.task_loaders import *
import dreamcoder.domains.re2.makeRe2Tasks as re2_legacy
from dreamcoder.domains.re2.re2Primitives import tfullstr

DOMAIN_NAME = "re2"
ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/{DOMAIN_NAME}")
DEFAULT_DATASET = f"{DOMAIN_NAME}_500"
DEFAULT_TASKS_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS)
DEFAULT_LANGUAGE_DIRECTORY = os.path.join(DEFAULT_DATA_DIRECTORY, LANGUAGE)


@TaskLoaderRegistry.register
class Re2Loader(TaskDataLoader):
    name = DOMAIN_NAME

    def load_tasks(self):
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, self.name)

        train_tasks, test_tasks = re2_legacy.loadRe2Dataset(
            task_dataset=DEFAULT_DATASET,
            task_dataset_dir=DEFAULT_TASKS_DIRECTORY,
            type_request=str(tfullstr),
        )

        tasks = {TRAIN: train_tasks, TEST: test_tasks}
        for split in tasks.keys():
            # Need to add supervision.
            pass
            # split_path = os.path.join(dataset_path, split)
            # task_files = sorted(
            #     [
            #         os.path.join(split_path, t)
            #         for t in os.listdir(split_path)
            #         if ".p" in t
            #     ],
            #     key=lambda p: os.path.basename(p),
            # )
            # for task_file in task_files:
            #     with open(task_file, "rb") as f:
            #         t = dill.load(f)
            #         t.nearest_name = None
            #         # Add the task serializer.
            #         t.ocaml_serializer = None
            #         # Add supervision.
            #         t.supervisedSolution = t.groundTruthProgram
            #         tasks[split].append(t)
        return tasks
