"""
experiment_iterator.py | Author : Catherine Wong.
Utility classes for initializing and running iterated experiments from configs.
"""
import copy
import json
import os

import numpy as np

import src.models.model_loaders as model_loaders
import src.task_loaders as task_loaders
import src.utils as utils
from dreamcoder.frontier import Frontier

# Experiment state config constants
METADATA = "metadata"
CURR_ITERATION = "curr_iteration"
EXPERIMENT_ID = "experiment_id"
EXPORT_DIRECTORY = "export_directory"
LOG_DIRECTORY = "log_directory"
RESUME_CHECKPOINT_DIRECTORY = "resume_checkpoint_directory"
EXPORT_WITH_TIMESTAMP = "export_with_timestamp"

TASKS_LOADER = "tasks_loader"
TASK_LANGUAGE_LOADER = "task_language_loader"
INIT_FRONTIERS_FROM_CHECKPOINT = "init_frontiers_from_checkpoint"
FRONTIERS_CHECKPOINT = "frontiers.json"
SAMPLES_CHECKPOINT = "samples.json"

MODEL_INITIALIZERS = "model_initializers"
MODEL_TYPE = "model_type"
MODEL_LOADER = "model_loader"
MODEL_INITIALIZER_FN = "model_initializer_fn"
PARAMS = "params"

TIMESTAMP = "timestamp"
TIMESTAMPED_EXPERIMENT_ID = "timestamped_experiment_id"
OCAML_SPECIAL_HANDLER = "ocaml_special_handler"
RANDOM_SEED = "random_seed"
RANDOM_GENERATOR = "random_generator"


LOG_DEBUG, LOG_WARNING, LOG_INFO = 3, 2, 1


class ExperimentState:

    ALL = "all"
    SAMPLES = "samples"
    FRONTIERS = "frontiers"

    def __init__(self, config):
        self.config = config

        self.metadata = self.init_metadata_from_config(config)
        self.curr_iteration = self.init_curr_iteration()

        self.tasks, self.task_frontiers = self.init_tasks_from_config(config)
        self.task_language, self.task_vocab = self.init_task_language_from_config(
            config
        )

        # Contains tasks, frontiers, language sampled from a generative model.
        self.sample_tasks = {split: [] for split in self.tasks.keys()}
        self.sample_language = {split: {} for split in self.tasks.keys()}
        self.sample_frontiers = {split: {} for split in self.tasks.keys()}

        self.models = {}
        self.init_models_from_config(config)

        self.init_log_and_export_from_config()

        self.maybe_resume_from_checkpoint()

    def init_tasks_from_config(self, config):
        task_loader = task_loaders.TaskLoaderRegistry[config[METADATA][TASKS_LOADER]]
        self.tasks = task_loader.load_tasks()

        self.task_frontiers = {
            split: {task: Frontier([], task=task) for task in self.tasks[split]}
            for split in self.tasks.keys()
        }

        return self.tasks, self.task_frontiers

    def init_task_language_from_config(self, config):
        language_loader = task_loaders.TaskLanguageLoaderRegistry[
            config[METADATA][TASK_LANGUAGE_LOADER]
        ]

        return language_loader.load_task_language()

    def init_models_from_config(self, config, models_to_initialize=None):
        for model_initializer_block in config[MODEL_INITIALIZERS]:
            model_type = model_initializer_block[MODEL_TYPE]
            if (
                models_to_initialize is not None
                and model_type not in models_to_initialize
            ):
                continue

            model_loader_registry = model_loaders.ModelLoaderRegistries[model_type]
            model_loader = model_loader_registry[model_initializer_block[MODEL_LOADER]]

            model_loader_fn = getattr(
                model_loader, model_initializer_block[MODEL_INITIALIZER_FN]
            )

            model = model_loader_fn(
                experiment_state=self, **model_initializer_block[PARAMS]
            )
            self.models[model_type] = model

    def init_metadata_from_config(self, config):
        metadata = copy.copy(config[METADATA])
        metadata[TIMESTAMP] = (
            utils.escaped_timestamp() if metadata[EXPORT_WITH_TIMESTAMP] else ""
        )
        metadata[TIMESTAMPED_EXPERIMENT_ID] = (
            f"{metadata[EXPERIMENT_ID]}_{metadata[TIMESTAMP]}"
            if metadata[EXPORT_WITH_TIMESTAMP]
            else metadata[EXPERIMENT_ID]
        )

        if RANDOM_SEED in metadata:
            metadata[RANDOM_GENERATOR] = np.random.default_rng(metadata[RANDOM_SEED])
        else:
            metadata[RANDOM_GENERATOR] = np.random.default_rng()
        return metadata

    def init_curr_iteration(self):
        return self.metadata.get(CURR_ITERATION, 0)

    def init_log_and_export_from_config(self):
        """Initializes time-stamped checkpoint directory and log file if log and export directory are provided."""

        log_directory = self.metadata[LOG_DIRECTORY]
        export_directory = self.metadata[EXPORT_DIRECTORY]

        if log_directory is not None:
            utils.mkdir_if_necessary(self.metadata[LOG_DIRECTORY])
            # Set log directory to the timestamped output file
            self.metadata[LOG_DIRECTORY] = os.path.join(
                self.metadata[LOG_DIRECTORY],
                self.metadata[TIMESTAMPED_EXPERIMENT_ID],
            )
            self.init_logger()

        if export_directory is not None:
            self.metadata[EXPORT_DIRECTORY] = os.path.join(
                self.metadata[EXPORT_DIRECTORY],
                self.metadata[TIMESTAMPED_EXPERIMENT_ID],
            )
            utils.mkdir_if_necessary(self.metadata[EXPORT_DIRECTORY])

    def init_logger(self):
        pass

    def log_metadata(self, verbosity=LOG_DEBUG):
        if verbosity >= LOG_DEBUG:
            keys_to_log = {k for k in self.metadata}
        elif verbosity < LOG_DEBUG and verbosity >= LOG_WARNING:
            keys_to_log = {}

        print(f"============LOGGING METADATA============")
        print(f"\tcurr_iteration: {self.curr_iteration}")
        for attr in keys_to_log:
            if attr in self.metadata:
                print(f"\t{attr}: {self.metadata[attr]}")
        print(f"====================================")

    def log_frontiers(self, verbosity=LOG_DEBUG, include_samples=False):
        print(f"============LOGGING FRONTIERS============")
        for task_split in self.task_frontiers:
            num_solved = len(self.get_non_empty_frontiers_for_split(task_split))
            print(
                f"\t total_solved_tasks_{task_split} @ iteration {self.curr_iteration}: {num_solved} / {len(self.task_frontiers[task_split])}"
            )

        if include_samples:
            for task_split in self.task_frontiers:
                sample_frontiers_non_empty = [
                    self.sample_frontiers[task_split][task]
                    for task in self.sample_frontiers[task_split]
                    if not self.sample_frontiers[task_split][task].empty
                ]
                print(
                    f"\t total_solved_sample_tasks_{task_split} @ iteration {self.curr_iteration}: {len(sample_frontiers_non_empty)} / {len(self.sample_frontiers[task_split])}"
                )
        print(f"====================================")

    def maybe_resume_from_checkpoint(self):
        if self.metadata[INIT_FRONTIERS_FROM_CHECKPOINT]:
            use_resume_checkpoint = (
                self.metadata[RESUME_CHECKPOINT_DIRECTORY] is not None
            )
            frontiers_loaded = self.load_frontiers_from_checkpoint(
                use_resume_checkpoint=use_resume_checkpoint
            )
            return frontiers_loaded

    def get_checkpoint_directory(self):
        checkpoint_directory = os.path.join(
            self.metadata[EXPORT_DIRECTORY], str(self.curr_iteration)
        )
        utils.mkdir_if_necessary(checkpoint_directory)
        return checkpoint_directory

    def get_resume_checkpoint_directory(self):
        return os.path.join(
            self.metadata[RESUME_CHECKPOINT_DIRECTORY], str(self.curr_iteration)
        )

    def checkpoint_frontiers(self):
        json_frontiers = {
            split: {
                task.name: self.task_frontiers[split][task].json()
                for task in self.task_frontiers[split]
            }
            for split in self.task_frontiers
        }
        checkpoint_directory = os.path.join(
            self.get_checkpoint_directory(), FRONTIERS_CHECKPOINT
        )
        with open(checkpoint_directory, "w") as f:
            json.dump(json_frontiers, f, indent=4)
        print(
            f"============Checkpointing frontiers to {checkpoint_directory}==========="
        )

    def load_frontiers_from_checkpoint(self, use_resume_checkpoint=False):
        """Note that this loads NON-EMPTY frontiers to combine with existing frontiers."""

        checkpoint_dir = (
            self.get_checkpoint_directory()
            if not use_resume_checkpoint
            else self.get_resume_checkpoint_directory()
        )
        frontiers_checkpoint = os.path.join(checkpoint_dir, FRONTIERS_CHECKPOINT)
        if not os.path.exists(frontiers_checkpoint):
            print(
                f"load_frontiers_from_checkpoint: No checkpoint found at: {frontiers_checkpoint}"
            )
            return False

        with open(frontiers_checkpoint, "r") as f:
            json_frontiers = json.load(f)

        for split in self.task_frontiers:
            for task in self.task_frontiers[split]:
                if task.name in json_frontiers[split]:
                    json_frontier = json_frontiers[split][task.name]
                    loaded_frontier = Frontier.from_json(
                        task, self.models[model_loaders.GRAMMAR], json_frontier
                    )
                    self.task_frontiers[split][task] = self.task_frontiers[split][
                        task
                    ].combine(loaded_frontier)
        f"============Loaded previously checkpointed frontiers from {frontiers_checkpoint}==========="
        return True

    def load_samples_from_checkpoint(self, use_resume_checkpoint=False):
        """NOTE(gg): This is currently untested and unused.

        CodexSampleGenerator already has the ability to cache and load queries via `use_cached`,
        so there is no current need to load samples from checkpoint.
        """
        checkpoint_dir = (
            self.get_checkpoint_directory()
            if not use_resume_checkpoint
            else self.get_resume_checkpoint_directory()
        )
        samples_checkpoint = os.path.join(checkpoint_dir, SAMPLES_CHECKPOINT)
        if not os.path.exists(samples_checkpoint):
            print(
                f"load_samples_from_checkpoint: No checkpoint found at: {samples_checkpoint}"
            )
            return False

        with open(samples_checkpoint, "r") as f:
            json_samples = json.load(f)

        # Update the tasks
        current_sample_tasks = {
            split: {t.name: t for t in self.sample_tasks[split]}
            for split in self.sample_tasks
        }
        for split in json_samples["tasks"]:
            for loaded_task in json_samples["tasks"][split]:
                current_sample_tasks[split][loaded_task.name] = loaded_task.from_json()
        self.sample_tasks = {
            split: list(current_sample_tasks[split].values())
            for split in current_sample_tasks
        }

        # Update the frontiers
        for split in json_samples["frontiers"]:
            for task in json_samples["frontiers"][split]:
                loaded_frontier = Frontier.from_json(
                    task,
                    self.models[model_loaders.GRAMMAR],
                    json_samples["frontiers"][split][task],
                )
                if task.name in self.sample_frontiers[split]:
                    self.sample_frontiers[split][task] = self.sample_frontiers[split][
                        task
                    ].combine(loaded_frontier)
                else:
                    self.sample_frontiers[split][task] = loaded_frontier

        # Update the language
        for split in json_samples["language"]:
            for task in json_samples["language"][split]:
                self.sample_language[split][task] = json_samples["language"][split][
                    task
                ]

        print(
            f"============Loaded previously checkpointed samples from {samples_checkpoint}==========="
        )
        return True

    def checkpoint_samples(self):
        """Save sample_tasks, sample_frontiers, and sample_language to JSON."""
        sample_tasks = {
            split: {task.name: task.json() for task in self.sample_tasks[split]}
            for split in self.sample_tasks
        }
        sample_frontiers = {
            split: {
                task.name: self.sample_frontiers[split][task].json()
                for task in self.sample_frontiers[split]
            }
            for split in self.sample_frontiers
        }
        sample_language = {
            split: {
                task.name: self.sample_language[split][task].json()
                for task in self.sample_language[split]
            }
            for split in self.sample_language
        }

        checkpoint_directory = os.path.join(
            self.get_checkpoint_directory(), SAMPLES_CHECKPOINT
        )
        with open(checkpoint_directory, "w") as f:
            json.dump(
                {
                    "sample_tasks": sample_tasks,
                    "sample_frontiers": sample_frontiers,
                    "sample_language": sample_language,
                },
                f,
                indent=4,
            )
        print(f"============Checkpointing samples to {checkpoint_directory}===========")

    def checkpoint_state(self, state_to_checkpoint):
        for state in state_to_checkpoint:
            if state == self.FRONTIERS:
                self.checkpoint_frontiers()
            elif state == self.SAMPLES:
                self.checkpoint_samples()
            else:
                raise NotImplementedError

    def checkpoint_models(self, models_to_checkpoint):
        for model in models_to_checkpoint:
            if model in self.models:
                self.models[model].checkpoint(self, self.get_checkpoint_directory())

    def load_models_from_checkpoint(self):
        pass

    def get_non_empty_frontiers_for_split(self, task_split):
        """Returns the non empty frontiers within a split."""
        return [
            self.task_frontiers[task_split][task]
            for task in self.task_frontiers[task_split]
            if not self.task_frontiers[task_split][task].empty
        ]

    def get_language_for_ids(
        self,
        task_split,
        task_ids,
        include_samples=False,
        include_ground_truth_tasks=True,
    ):
        """Returns array of language for list of task_ids. If task_ids is ALL, returns all tasks in task_split and does NOT return samples."""
        language = [
            self.task_language[task_split][task.name]
            for task in self.get_tasks_for_ids(
                task_split,
                task_ids,
                include_samples,
                include_ground_truth_tasks,
            )
            if task.name in self.task_language[task_split]
        ]
        if include_samples:
            language += self.sample_language[task_split]
        return language

    def get_tasks_for_ids(
        self,
        task_split,
        task_ids,
        include_samples=False,
        include_ground_truth_tasks=True,
    ):
        """Returns array of tasks for list of task_ids. If task_ids is ALL, returns all tasks in task_split and does NOT return samples."""
        tasks = []
        if include_ground_truth_tasks:
            if task_ids == self.ALL:
                tasks = [t for t in self.tasks[task_split]]
            else:
                tasks = [t for t in self.tasks[task_split] if t.name in task_ids]
        if include_samples:
            # Include all of the samples for a given split.
            tasks += self.sample_tasks[task_split]
        return tasks

    def get_frontiers_for_ids(
        self,
        task_split,
        task_ids,
        include_samples=False,
        include_ground_truth_tasks=True,
    ):
        """Returns array of frontiers for list of task_ids. Indicate whether to include samples or regular frontiers."""
        frontiers = [
            self.task_frontiers[task_split][task]
            for task in self.get_tasks_for_ids(
                task_split,
                task_ids,
                include_samples,
                include_ground_truth_tasks,
            )
            if task in self.task_frontiers[task_split]
        ]
        if include_samples:
            frontiers += list(self.sample_frontiers[task_split].values())
        return frontiers

    def get_frontiers_for_ids_in_splits(
        self,
        task_splits,
        task_ids_in_splits,
        include_samples=False,
        include_ground_truth_tasks=True,
    ):
        """Returns {split: [array of frontiers]} for [split in task_splits] and task_ids_in_splits = {split : [task_ids or ALL]}. Indicate whether to include samples or regular frontiers."""
        return {
            task_split: self.get_frontiers_for_ids(
                task_split,
                task_ids_in_splits[task_split],
                include_samples,
                include_ground_truth_tasks,
            )
            for task_split in task_splits
        }

    def update_frontiers(self, new_frontiers, maximum_frontier, task_split, is_sample):
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

    def initialize_ground_truth_task_frontiers(
        self, task_split, exclude_nonempty=True, compute_likelihoods=False
    ):
        """Updates frontiers to ground truth programs. Expects the ground truth programs to be under task.supervision"""
        for task in self.task_frontiers[task_split]:
            if exclude_nonempty:
                if not self.task_frontiers[task_split][task].empty:
                    continue

            self.task_frontiers[task_split][task] = self.task_frontiers[task_split][
                task
            ].replaceWithSupervised(
                g=self.models[model_loaders.GRAMMAR],
                compute_likelihoods=compute_likelihoods,
            )

    def reset_task_frontiers(self, task_split, task_ids):
        tasks = self.get_tasks_for_ids(
            task_split,
            task_ids,
            include_samples=False,
            include_ground_truth_tasks=True,
        )
        for task in tasks:
            self.task_frontiers[task_split][task] = Frontier.makeEmpty(task)
        print(f"reset_task_frontiers for split: {task_split}")

    def reset_samples(self, task_split):
        self.sample_tasks[task_split] = []
        self.sample_language[task_split] = {}
        self.sample_frontiers[task_split] = {}
        print(f"reset_samples for split: {task_split}")


# Experiment iterator config constants
EXPERIMENT_ITERATOR = "experiment_iterator"
MAX_ITERATIONS = "max_iterations"
TASK_BATCHER = "task_batcher"
LOOP_BLOCKS = "loop_blocks"
EXPERIMENT_BLOCK_TYPE = "experiment_block_type"
EXPERIMENT_BLOCK_TYPE_MODEL_FN = "model_fn"  # Run a function from a model
EXPERIMENT_BLOCK_TYPE_CHECKPOINT = "checkpoint"  # Checkpoint the model state
EXPERIMENT_BLOCK_TYPE_STATE_FN = "state_fn"
STATE_TO_CHECKPOINT = "state_to_checkpoint"
MODELS_TO_CHECKPOINT = "models_to_checkpoint"


class ExperimentIterator:
    def __init__(self, config, experiment_state):
        self.config = config

        (
            self.curr_iteration,
            self.max_iterations,
            self.task_batcher,
            self.loop_pointer,
            self.loop_blocks,
        ) = self.init_iterator_from_config(config, experiment_state)

    def init_iterator_from_config(self, config, experiment_state):
        max_iterations = config[EXPERIMENT_ITERATOR][MAX_ITERATIONS]

        if experiment_state.curr_iteration is None:
            curr_iteration = 0
            experiment_state.curr_iteration = curr_iteration
        else:
            curr_iteration = experiment_state.curr_iteration

        task_batcher = task_loaders.TaskBatcherRegistry.get(
            config[EXPERIMENT_ITERATOR][TASK_BATCHER][MODEL_TYPE],
            experiment_state=experiment_state,
            curr_iteration=curr_iteration,
            max_iterations=max_iterations,
            **config[EXPERIMENT_ITERATOR][TASK_BATCHER][PARAMS],
        )

        loop_pointer = 0
        loop_blocks = config[EXPERIMENT_ITERATOR][LOOP_BLOCKS]

        return (
            curr_iteration,
            max_iterations,
            task_batcher,
            loop_pointer,
            loop_blocks,
        )

    def is_finished(self):
        return self.curr_iteration >= self.max_iterations

    def next(self, experiment_state):
        """Increment the iterator. Currently supports the following types of experiment blocks:
        model_fn: run a model function on a batch of tasks.
        checkpoint: checkpoint state or models.
        """
        curr_loop_block = self.loop_blocks[self.loop_pointer]
        if curr_loop_block[EXPERIMENT_BLOCK_TYPE] == EXPERIMENT_BLOCK_TYPE_MODEL_FN:
            self.execute_model_fn(experiment_state, curr_loop_block)
        elif curr_loop_block[EXPERIMENT_BLOCK_TYPE] == EXPERIMENT_BLOCK_TYPE_STATE_FN:
            self.execute_state_fn(experiment_state, curr_loop_block)
        elif curr_loop_block[EXPERIMENT_BLOCK_TYPE] == EXPERIMENT_BLOCK_TYPE_CHECKPOINT:
            self.checkpoint(experiment_state, curr_loop_block)

        self.loop_pointer += 1

        if self.loop_pointer >= len(self.loop_blocks):
            self.loop_pointer = self.loop_pointer % len(self.loop_blocks)
            self.curr_iteration += 1
            experiment_state.curr_iteration = self.curr_iteration

    def execute_state_fn(self, experiment_state, curr_loop_block):
        """Executes a function on the experiment state."""
        state_fn_name = curr_loop_block[EXPERIMENT_BLOCK_TYPE_STATE_FN]
        state_fn = getattr(experiment_state, state_fn_name)

        state_fn(
            **curr_loop_block[PARAMS],
        )

    def log_model_fn(self, experiment_state, curr_loop_block, task_batch_ids):
        print(f"============LOGGING MODEL_FN============")

        keys_to_log = {k for k in curr_loop_block}
        for attr in keys_to_log:
            if attr in curr_loop_block:
                print(f"\t{attr}: {curr_loop_block[attr]}")

        if type(task_batch_ids) == list:
            print(
                f"task_ids: {len(task_batch_ids)} tasks: {task_batch_ids[0]} -- {task_batch_ids[-1]}"
            )
        else:
            for split in task_batch_ids:
                print(
                    f"task_ids {split}: {len(task_batch_ids[split])} tasks: {task_batch_ids[split][0]} -- {task_batch_ids[split][-1]}"
                )
        print(f"====================================")

    def execute_model_fn(self, experiment_state, curr_loop_block):
        """Executes a model function on a batch of tasks. Model functions should be of the form:

        model_fn(experiment_state, task_split, task_batch_ids, **params...)

        OR
        model_fn(experiment_state, task_splits, task_ids_in_splits, **params...)
        """
        if task_loaders.TASK_SPLIT in curr_loop_block:
            self.execute_model_fn_single_split(experiment_state, curr_loop_block)
        elif task_loaders.TASK_SPLITS in curr_loop_block:
            self.execute_model_fn_several_splits(experiment_state, curr_loop_block)
        else:
            raise ValueError

    def execute_model_fn_several_splits(self, experiment_state, curr_loop_block):
        task_splits, task_batch_sizes = (
            curr_loop_block[task_loaders.TASK_SPLITS],
            curr_loop_block[task_loaders.TASK_BATCH_SIZES],
        )
        task_ids_in_splits = {
            task_split: self.task_batcher.get_task_batch_ids(
                experiment_state, self.curr_iteration, task_split, task_batch_size
            )
            for task_split, task_batch_size, in zip(task_splits, task_batch_sizes)
        }

        # Log the model function.
        self.log_model_fn(experiment_state, curr_loop_block, task_ids_in_splits)

        # Run model function on the batch of tasks
        model_type, model_fn_name = (
            curr_loop_block[MODEL_TYPE],
            curr_loop_block[EXPERIMENT_BLOCK_TYPE_MODEL_FN],
        )
        model_fn = getattr(experiment_state.models[model_type], model_fn_name)

        model_fn(
            experiment_state=experiment_state,
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            **curr_loop_block[PARAMS],
        )

    def execute_model_fn_single_split(self, experiment_state, curr_loop_block):
        # Get a batch of tasks
        task_split, task_batch_size = (
            curr_loop_block[task_loaders.TASK_SPLIT],
            curr_loop_block[task_loaders.TASK_BATCH_SIZE],
        )
        task_batch_ids = self.task_batcher.get_task_batch_ids(
            experiment_state, self.curr_iteration, task_split, task_batch_size
        )

        # Log the model function.
        self.log_model_fn(experiment_state, curr_loop_block, task_batch_ids)

        # Run model function on the batch of tasks
        model_type, model_fn_name = (
            curr_loop_block[MODEL_TYPE],
            curr_loop_block[EXPERIMENT_BLOCK_TYPE_MODEL_FN],
        )
        model_fn = getattr(experiment_state.models[model_type], model_fn_name)

        model_fn(
            experiment_state=experiment_state,
            task_split=task_split,
            task_batch_ids=task_batch_ids,
            **curr_loop_block[PARAMS],
        )

    def checkpoint(self, experiment_state, curr_loop_block):
        experiment_state.checkpoint_state(curr_loop_block[STATE_TO_CHECKPOINT])
        experiment_state.checkpoint_models(curr_loop_block[MODELS_TO_CHECKPOINT])
