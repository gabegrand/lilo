"""
make_tasks.py | Author: Catherine Wong
Loads tasks and language for the CLEVR domain.
"""

from src.task_loaders import *
from data.clevr.make_ground_truth_programs import *
import dreamcoder.domains.clevr.makeClevrTasks as makeClevrTasks

DOMAIN_NAME = "clevr"

ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/{DOMAIN_NAME}")

DEFAULT_TASK_DATASET_DIR = DEFAULT_DATA_DIRECTORY
DEFAULT_LANGUAGE_DIR = os.path.join(DEFAULT_DATA_DIRECTORY, "language")


@TaskLoaderRegistry.register
class ClevrLoader(TaskDataLoader):
    name = DOMAIN_NAME

    def load_tasks(self):
        args = {
            "curriculumDatasets": [],
            "taskDatasets": ["all"],
            "taskDatasetDir": DEFAULT_TASK_DATASET_DIR,
            "languageDatasetDir": DEFAULT_LANGUAGE_DIR,
        }
        (
            train_tasks,
            test_tasks,
            language_dataset,
        ) = makeClevrTasks.loadAllTaskAndLanguageDatasets(args)
        tasks = {TRAIN: train_tasks, TEST: test_tasks}

        for split in tasks.keys():
            for t in tasks[split]:
                t.supervisedSolution = make_ground_truth_program_for_task(t)
                t.groundTruthProgram = t.supervisedSolution
        return tasks
