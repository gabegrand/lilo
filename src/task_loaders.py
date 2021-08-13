"""
task_loaders.py | Author : Catherine Wong
Utility classes for loading and batching datasets of tasks and language.
"""

import os
import json
from class_registry import ClassRegistry

TaskLoaderRegistry = ClassRegistry("name", unique=True)
TaskLanguageLoaderRegistry = ClassRegistry("name", unique=True)
TaskBatcherRegistry = ClassRegistry("name", unique=True)

# Task and dataset constants
TRAIN, TEST = "train", "test"
HUMAN, SYNTHETIC = "human", "synthetic"
TASKS, LANGUAGE, VOCAB = "tasks", "language", "vocab"

TASK_SPLIT, TASK_BATCH_SIZE = "task_split", "task_batch_size"


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


class TaskBatcher:
    """Abstract class for task batchers."""

    def __init__(self, experiment_state, curr_iteration, max_iterations):
        raise NotImplementedError

    def get_task_batch_ids(
        self, experiment_state, curr_iteration, task_split, batch_size
    ):
        raise NotImplementedError
