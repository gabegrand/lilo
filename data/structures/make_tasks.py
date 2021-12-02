"""
structures: make_tasks.py
Author: Catherine Wong.

Utility functions for loading tasks and language for the structures domain. This domain was collected by McCarthy et. al.
"""
import os
import dill
import pandas as pd

from src.task_loaders import *

from dreamcoder.domains.tower.makeTowerTasks import SupervisedTower

SUBDOMAINS = ["bridge", "castle", "house", "city"]
TASKS_URL = "https://lax-structures-{}-all.s3.amazonaws.com/df_{}.csv"

PICKLE_SUFFIX = ".p"


ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, "data/structures")


@TaskLoaderRegistry.register
class Structures1KLoader(TaskDataLoader):
    name = "structures_1K"

    def load_tasks(self):
        tasks = {TRAIN: [], TEST: []}
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, self.name)
        for split in tasks.keys():
            split_path = os.path.join(dataset_path, split)
            if not (
                os.path.exists(split_path)
                and len([f for f in os.listdir(split_path) if PICKLE_SUFFIX in f]) > 0
            ):
                for subdomain in SUBDOMAINS:
                    self.make_structures_tasks(subdomain, dataset_path)

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
                    t = dill.load(f)
                    t.nearest_name = None
                    # Add the task serializer.
                    t.ocaml_serializer = None
                    # Add supervision.
                    t.supervisedSolution = t.original
                    tasks[split].append(t)
        return tasks

    def make_structures_tasks(self, subdomain, dataset_path):
        import pathlib

        pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        for split in TRAIN, TEST:
            pathlib.Path(os.path.join(dataset_path, split)).mkdir(
                parents=True, exist_ok=True
            )
        # Pull the task CSV
        df = pd.read_csv(TASKS_URL.format(subdomain, subdomain))

        def get_split(is_train):
            return TRAIN if is_train else TEST

        for _, row in df.iterrows():
            structure_type, structure_number, program, split = (
                row["structure_type"],
                row["structure_number"],
                row["dreamcoder_program"],
                get_split(row["train"]),
            )
            # McCarthy et. al use 2x1 and 1x2 blocks, unlike the DC 3x1 and 1x3 blocks
            program = program.replace("h", "2x1")
            program = program.replace("t", "1x2")
            task_name = f"{structure_type}_{structure_number}"
            task = SupervisedTower(name=task_name, program=program)
            task_file = os.path.join(dataset_path, split, f"{task_name}.p")
            with open(task_file, "wb") as f:
                dill.dump(task, f)
