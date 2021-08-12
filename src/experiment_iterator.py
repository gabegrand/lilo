"""
experiment_iterator.py | Author : Catherine Wong.
Utility classes for initializing and running iterated experiments from configs.
"""
import os, json
from class_registry import ClassRegistry

from dreamcoder.frontier import Frontier

import src.utils as utils

# Registries for dataset and models
TaskLoaderRegistry = ClassRegistry("name", unique=True)
TaskLanguageLoaderRegistry = ClassRegistry("name", unique=True)

GRAMMAR = "grammar"
ModelLoaderRegistries = {GRAMMAR: ClassRegistry("name", unique=True)}

# Task and dataset constants
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


# Experiment config constants
METADATA = "metadata"
TASKS_LOADER = "tasks_loader"
TASK_LANGUAGE_LOADER = "task_language_loader"
INIT_FRONTIERS_FROM_CHECKPOINT = "init_frontiers_from_checkpoint"

MODEL_INITIALIZERS = "model_initializers"
MODEL_TYPE = "model_type"
MODEL_LOADER = "model_loader"
MODEL_INITIALIZER_FN = "model_initializer_fn"
PARAMS = "params"

TIMESTAMP = "timestamp"
OCAML_SPECIAL_HANDLER = "ocaml_special_handler"


class ExperimentState:
    ALL = "all"

    def __init__(self, config):
        self.tasks, self.task_frontiers = self.init_tasks_from_config(config)
        self.task_language, self.task_vocab = self.init_task_language_from_config(
            config
        )
        self.sample_tasks, self.sample_frontiers, self.sample_language = {}, {}, {}

        self.models = {}
        self.init_models_from_config(config)

        self.metadata = self.init_metadata_from_config(config)

    def init_tasks_from_config(self, config):
        task_loader = TaskLoaderRegistry[config[METADATA][TASKS_LOADER]]
        tasks = task_loader.load_tasks()

        # TODO: allow initialization from frontiers.
        task_frontiers = {
            split: {task: Frontier([], task=task) for task in tasks[split]}
            for split in tasks.keys()
        }

        if config[METADATA][INIT_FRONTIERS_FROM_CHECKPOINT]:
            raise NotImplementedError

        return tasks, task_frontiers

    def init_task_language_from_config(self, config):
        language_loader = TaskLanguageLoaderRegistry[
            config[METADATA][TASK_LANGUAGE_LOADER]
        ]

        return language_loader.load_task_language()

    def init_models_from_config(self, config):
        for model_initializer_block in config[MODEL_INITIALIZERS]:
            model_type = model_initializer_block[MODEL_TYPE]
            model_loader_registry = ModelLoaderRegistries[model_type]
            model_loader = model_loader_registry[
                model_initializer_block[MODEL_LOADER]
            ]

            model_loader_fn = getattr(
                model_loader, model_initializer_block[MODEL_INITIALIZER_FN]
            )

            model = model_loader_fn(**model_initializer_block[PARAMS])
            self.models[model_type] = model

    def init_metadata_from_config(self, config):
        metadata = config[METADATA]
        metadata[TIMESTAMP] = utils.escaped_timestamp()
        return metadata

    def checkpoint_frontiers(self):
        pass

    def checkpoint_samples(self):
        pass

    def checkpoint_models(self):
        pass

    def get_tasks_for_ids(self, task_split, task_ids, include_samples=True):
        """Returns array of tasks for list of task_ids. If task_ids is ALL, returns all tasks in task_split and does NOT return samples."""

        if task_ids == self.ALL:
            return self.tasks[task_split]

        tasks = [t for t in self.tasks[task_split] if t.name in task_ids]
        if include_samples:
            tasks += [t for t in self.sample_tasks if t.name in task_ids]
        return tasks

    def get_frontiers_for_ids(self, task_split, task_ids, include_samples=False):
        """Returns array of frontiers for list of task_ids. If task_ids is ALL, returns frontiers for all tasks in task_split and does NOT return samples."""
        return [
            self.task_frontiers[task_split][task]
            for task in self.get_tasks_for_ids(task_split, task_ids, include_samples)
        ]

    def update_frontiers(
        self, new_frontiers, maximum_frontier, task_split, is_sample
    ):
        """Updates frontiers with new_frontiers. If is_sample, updates sample frontiers."""

        for new_frontier in new_frontiers:
            if is_sample:
                if new_frontier.task in self.sample_frontiers:
                    self.sample_frontiers[new_frontier.task] = (
                        self.sample_frontiers[new_frontier.task]
                        .combine(new_frontier)
                        .topK(maximum_frontier)
                    )
            else:
                if new_frontier.task in self.task_frontiers[task_split]:
                    self.task_frontiers[task_split][new_frontier.task] = (
                        self.task_frontiers[task_split][new_frontier.task]
                        .combine(new_frontier)
                        .topK(maximum_frontier)
                    )
