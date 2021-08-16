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
        self.random_seed = None
        self.task_id_orderings = {}
        self.batch_pointers = {}

    def get_task_batch_ids(
        self, experiment_state, curr_iteration, task_split, batch_size
    ):
        raise NotImplementedError


@TaskBatcherRegistry.register
class OrderedTaskBatcher(TaskBatcher):
    """OrderedTaskBatcher: gets batch_size ids with wraparound. No shuffle."""

    name = "ordered_task_batcher"

    def __init__(self, experiment_state, curr_iteration, max_iterations):
        self.task_id_orderings = {
            split: [t.name for t in experiment_state.tasks[split]]
            for split in experiment_state.tasks
        }
        self.batch_pointers = {split: 0 for split in experiment_state.tasks}

    def get_task_batch_ids(
        self, experiment_state, curr_iteration, task_split, batch_size
    ):
        all_tasks_for_split = self.task_id_orderings[task_split]
        start = self.batch_pointers[task_split] % len(all_tasks_for_split)

        end = start + batch_size

        task_batch = (all_tasks_for_split + all_tasks_for_split.copy())[
            start:end
        ]  # Wraparound nicely.

        self.batch_pointers[task_split] += batch_size

        return task_batch
