"""
make_tasks.py | Author: Catherine Wong
Loads tasks and language for the CLEVR domain.
"""

from src.task_loaders import *
from data.clevr.make_ground_truth_programs import *
import dreamcoder.domains.clevr.makeClevrTasks as makeClevrTasks
from dreamcoder.languageUtilities import languageForTasks

DOMAIN_NAME = "clevr"

ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/{DOMAIN_NAME}")

DEFAULT_TASK_DATASET_DIR = DEFAULT_DATA_DIRECTORY
DEFAULT_LANGUAGE_DIR = os.path.join(DEFAULT_DATA_DIRECTORY, "language")

clevr_dataset_keys = [
    "2_localization",
    "2_remove",
    "2_transform",
    "1_zero_hop",
    "1_single_or",
    "1_one_hop",
    "1_compare_integer",
]


class ClevrLoaderWrapper:
    def load_tasks_and_language(self, language_dataset_directory=DEFAULT_LANGUAGE_DIR):
        args = {
            "curriculumDatasets": [],
            "taskDatasets": clevr_dataset_keys,
            "taskDatasetDir": DEFAULT_TASK_DATASET_DIR,
            "languageDatasetDir": language_dataset_directory,
        }
        (
            train_tasks,
            test_tasks,
            language_dataset,
        ) = makeClevrTasks.loadAllTaskAndLanguageDatasets(args)
        tasks = {TRAIN: train_tasks, TEST: test_tasks}

        language, vocab = None, None
        if language_dataset_directory is not None:
            task_dict = {}
            for split in tasks:
                task_dict[split] = {}
                for t in tasks[split]:
                    task_dict[split][t.name] = []
            language, vocab = languageForTasks(
                languageDataset=language_dataset,
                languageDatasetDir=language_dataset_directory,
                taskDict=task_dict,
                use_splits=True,
            )
        return tasks, (language, vocab)


@TaskLoaderRegistry.register
class ClevrLoader(TaskDataLoader, ClevrLoaderWrapper):
    name = DOMAIN_NAME

    def load_tasks(self):
        tasks, _ = self.load_tasks_and_language(language_dataset_directory=None)

        for split in tasks.keys():
            supervised_tasks = []
            for t in tasks[split]:
                try:
                    t.supervisedSolution = make_ground_truth_program_for_task(t)
                    t.groundTruthProgram = t.supervisedSolution
                except:
                    print(f"Excluding task: {t.name}")
                    continue
                supervised_tasks.append(t)
            tasks[split] = supervised_tasks
        return tasks


@TaskLanguageLoaderRegistry.register
class ClevrHumanLanguageLoader(TaskDataLoader, ClevrLoaderWrapper):
    name = "clevr_human"

    def load_task_language(self):
        _, (language, vocab) = self.load_tasks_and_language(
            language_dataset_directory=DEFAULT_LANGUAGE_DIR + "_human"
        )
        return language, vocab


@TaskLanguageLoaderRegistry.register
class ClevrSyntheticLanguageLoader(TaskDataLoader, ClevrLoaderWrapper):
    name = "clevr_synthetic"

    def load_task_language(self):
        _, (language, vocab) = self.load_tasks_and_language(
            language_dataset_directory=DEFAULT_LANGUAGE_DIR
        )
        return language, vocab
