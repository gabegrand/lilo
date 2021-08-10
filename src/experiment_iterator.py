"""
experiment_iterator.py | Author : Catherine Wong.
Utility classes for initializing and running iterated experiments from configs.
"""
import os, json
from class_registry import ClassRegistry

TaskLoaderRegistry = ClassRegistry("name", unique=True)
TaskLanguageLoaderRegistry = ClassRegistry("name", unique=True)

TASKS, LANGUAGE, VOCAB = "tasks", "language", "vocab"
TRAIN, TEST = "train", "test"
HUMAN, SYNTHETIC = "human", "synthetic"


class TaskDataLoader:
    """Abstract class for Task and language dataset loaders."""

    def load_tasks(self):
        """:ret: {SPLIT : [array of tasks]}"""
        raise NotImplementedError

    def load_task_language(self):
        """:ret: {
        language: {split: task_name : [array of language]};
        vocab: {split: [array of tokens]}
        """
        raise NotImplementedError

    def load_task_language_for_directory(self, dataset_path, splits):
        language, vocab = {}, {}
        for split in splits:
            language_file = os.path.join(dataset_path, split, f"{LANGUAGE}.json")
            vocab_file = os.path.join(dataset_path, split, f"{VOCAB}.json")
            with open(language_file) as f:
                language[split] = json.load(f)
            with open(vocab_file) as f:
                vocab[split] = json.load(f)
        return language, vocab


class ExperimentState:
    def __init__(self, config):
        pass
