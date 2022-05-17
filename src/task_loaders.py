"""
task_loaders.py | Author : Catherine Wong
Utility classes for loading and batching datasets of tasks and language.
"""

import os
import json
import random
from class_registry import ClassRegistry

TaskLoaderRegistry = ClassRegistry("name", unique=True)
TaskLanguageLoaderRegistry = ClassRegistry("name", unique=True)
TaskBatcherRegistry = ClassRegistry("name", unique=True)

# Task and dataset constants
TRAIN, TEST = "train", "test"
HUMAN, SYNTHETIC = "human", "synthetic"
TASKS, PROGRAMS, LANGUAGE, VOCAB = (
    "tasks",
    "programs",
    "language",
    "vocab",
)

TASK_SPLIT, TASK_BATCH_SIZE = "task_split", "task_batch_size"
TASK_SPLITS, TASK_BATCH_SIZES = "task_splits", "task_batch_sizes"

ALL = "all"
GLOBAL_BATCH_SIZE = "global_batch_size"
RANDOM_SEED = "random_seed"


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

    def __init__(
        self,
        experiment_state,
        curr_iteration,
        max_iterations,
        global_batch_size,
        increment_at_global_iteration=True,
        verbose=False,
    ):
        self.increment_at_global_iteration = increment_at_global_iteration
        self.global_batch_size = global_batch_size
        self.task_id_orderings = {
            split: [t.name for t in experiment_state.tasks[split]]
            for split in experiment_state.tasks
        }
        if verbose:
            self.initial_log()

    def _subclass_initialization():
        pass

    def initial_log(self):
        print(f"============LOGGING TASK_BATCHER============")
        print(f"Task batcher: {self.name}")
        print(f"Initializing batcher over tasks: ")
        for split in self.task_id_orderings:
            print(f"{split} tasks: {len(self.task_id_orderings[split])}")
        print(f"global_batch_size: {self.global_batch_size}")
        print(f"====================================")

    def get_task_batch_ids(
        self, experiment_state, curr_iteration, task_split, batch_size
    ):
        """
        Gets a batch of tasks relative to a global pointer.
        batch_size: {ALL, GLOBAL_BATCH_SIZE, scalar}: 
        if ALL, returns all tasks for a split. 
        if GLOBAL_BATCH_SIZE, returns global_batch_size tasks from the global pointer.
        Else, returns n tasks from the global pointer.
        """
        all_tasks_for_split = self.task_id_orderings[task_split]
        if batch_size == ALL:
            return all_tasks_for_split

        # Calculate batching wrt. a global pointer.
        batch_size = (
            batch_size if batch_size != GLOBAL_BATCH_SIZE else self.global_batch_size
        )
        batch_size = int(batch_size)

        if self.increment_at_global_iteration:
            global_batch_start = self.global_batch_size * curr_iteration
        else:
            global_batch_start = 0

        start = global_batch_start % len(all_tasks_for_split)

        end = start + batch_size

        task_batch = (all_tasks_for_split + all_tasks_for_split.copy())[
            start:end
        ]  # Wraparound nicely.
        return task_batch


@TaskBatcherRegistry.register
class RandomShuffleOrderedTaskBatcher(OrderedTaskBatcher):
    """RandomShuffleOrderedTaskBatcher: shuffles tasks but maintains a shuffled ordered over epochs"""

    name = "random_shuffle_ordered_task_batcher"

    def __init__(
        self,
        experiment_state,
        curr_iteration,
        max_iterations,
        global_batch_size,
        increment_at_global_iteration,
        verbose=False,
    ):
        super(RandomShuffleOrderedTaskBatcher, self).__init__(
            experiment_state,
            curr_iteration,
            max_iterations,
            global_batch_size,
            increment_at_global_iteration,
            verbose,
        )
        self.seed = experiment_state.metadata[RANDOM_SEED]

        random.seed(self.seed)
        for split in self.task_id_orderings:
            if split != TEST:
                random.shuffle(self.task_id_orderings[split])
        print(f"random_seed: {self.seed}")


@TaskBatcherRegistry.register
class GroundTruthOrderedTaskBatcher(OrderedTaskBatcher):
    """GroundTruthOrderedTaskBatcher: orders tasks according to their description length in the reference ground truth. No shuffle."""

    name = "ground_truth_ordered_task_batcher"

    def __init__(
        self,
        experiment_state,
        curr_iteration,
        max_iterations,
        global_batch_size,
        increment_at_global_iteration,
        verbose=False,
    ):
        """
        increment_at_global_iteration: if True, uses a sliding pointer to return the next batch of global_batch_size with each global iteration of the LAPS experiment. If false, remains fixed at the first batch even at progressive iterations.
        """
        super(GroundTruthOrderedTaskBatcher, self).__init__(
            experiment_state,
            curr_iteration,
            max_iterations,
            global_batch_size,
            increment_at_global_iteration,
            verbose,
        )
        from dreamcoder.frontier import Frontier

        gt_frontiers = {
            task_split: {
                task: Frontier.makeFrontierFromSupervised(task)
                for task in experiment_state.task_frontiers[task_split]
            }
            for task_split in experiment_state.task_frontiers
        }

        def best_description_length(task, task_split):
            frontier = gt_frontiers[task_split][task]
            return min(
                [
                    len(e.program.left_order_tokens(show_vars=True))
                    for e in frontier.entries
                ]
            )

        def sorted_description_lengths(task_split):
            return sorted(
                gt_frontiers[task_split],
                key=lambda task: best_description_length(task, task_split),
            )

        self.task_id_orderings = {
            split: [t.name for t in sorted_description_lengths(split)]
            for split in gt_frontiers
        }
