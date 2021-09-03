"""
experiment_iterator.py | Author : Catherine Wong.
Utility classes for initializing and running iterated experiments from configs.
"""
import os, json

from dreamcoder.frontier import Frontier

import src.utils as utils
import src.models.model_loaders as model_loaders
import src.task_loaders as task_loaders

# Experiment state config constants
METADATA = "metadata"
EXPERIMENT_ID = "experiment_id"
EXPORT_DIRECTORY = "export_directory"
LOG_DIRECTORY = "log_directory"
RESUME_CHECKPOINT_DIRECTORY = "resume_checkpoint_directory"

TASKS_LOADER = "tasks_loader"
TASK_LANGUAGE_LOADER = "task_language_loader"
INIT_FRONTIERS_FROM_CHECKPOINT = "init_frontiers_from_checkpoint"

MODEL_INITIALIZERS = "model_initializers"
MODEL_TYPE = "model_type"
MODEL_LOADER = "model_loader"
MODEL_INITIALIZER_FN = "model_initializer_fn"
PARAMS = "params"

TIMESTAMP = "timestamp"
TIMESTAMPED_EXPERIMENT_ID = "timestamped_experiment_id"
OCAML_SPECIAL_HANDLER = "ocaml_special_handler"
RANDOM_SEED = "random_seed"


LOG_DEBUG, LOG_WARNING, LOG_INFO = 3, 2, 1


class ExperimentState:

    ALL = "all"
    SAMPLES = "samples"
    FRONTIERS = "frontiers"

    def __init__(self, config):
        self.tasks, self.task_frontiers = self.init_tasks_from_config(config)
        self.task_language, self.task_vocab = self.init_task_language_from_config(
            config
        )
        self.sample_tasks, self.sample_frontiers, self.sample_language = {}, {}, {}

        self.models = {}
        self.init_models_from_config(config)

        self.curr_iteration = None
        self.metadata = self.init_metadata_from_config(config)
        self.init_log_and_export_from_config()

    def init_tasks_from_config(self, config):
        task_loader = task_loaders.TaskLoaderRegistry[config[METADATA][TASKS_LOADER]]
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
        language_loader = task_loaders.TaskLanguageLoaderRegistry[
            config[METADATA][TASK_LANGUAGE_LOADER]
        ]

        return language_loader.load_task_language()

    def init_models_from_config(self, config):
        for model_initializer_block in config[MODEL_INITIALIZERS]:
            model_type = model_initializer_block[MODEL_TYPE]
            model_loader_registry = model_loaders.ModelLoaderRegistries[model_type]
            model_loader = model_loader_registry[model_initializer_block[MODEL_LOADER]]

            model_loader_fn = getattr(
                model_loader, model_initializer_block[MODEL_INITIALIZER_FN]
            )

            model = model_loader_fn(**model_initializer_block[PARAMS])
            self.models[model_type] = model

    def init_metadata_from_config(self, config):
        metadata = config[METADATA]
        metadata[TIMESTAMP] = utils.escaped_timestamp()
        metadata[
            TIMESTAMPED_EXPERIMENT_ID
        ] = f"{metadata[EXPERIMENT_ID]}_{metadata[TIMESTAMP]}"
        return metadata

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

    def log_frontiers(self, verbosity=LOG_DEBUG):
        print(f"============LOGGING FRONTIERS============")
        for task_split in self.task_frontiers:
            num_solved = len(self.get_non_empty_frontiers_for_split(task_split))
            print(
                f"\t total_solved_tasks_{task_split} @ iteration {self.curr_iteration}: {num_solved} / {len(self.task_frontiers[task_split])}"
            )
        print(f"====================================")

    def checkpoint_frontiers(self):
        pass

    def checkpoint_samples(self):
        pass

    def checkpoint_state(self, state_to_checkpoint):
        pass

    def checkpoint_models(self, models_to_checkpoint):
        pass

    def get_non_empty_frontiers_for_split(self, task_split):
        """Returns the non empty frontiers within a split."""
        return [
            self.task_frontiers[task_split][task]
            for task in self.task_frontiers[task_split]
            if not self.task_frontiers[task_split][task].empty
        ]

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
                return self.tasks[task_split]

            tasks = [t for t in self.tasks[task_split] if t.name in task_ids]
        if include_samples:
            if task_ids == self.ALL:
                return self.sample_tasks
            tasks += [t for t in self.sample_tasks if t.name in task_ids]
        return tasks

    def get_frontiers_for_ids(
        self,
        task_split,
        task_ids,
        include_samples=False,
        include_ground_truth_tasks=True,
    ):
        """Returns array of frontiers for list of task_ids. Indicate whether to include samples or regular frontiers."""
        return [
            self.task_frontiers[task_split][task]
            for task in self.get_tasks_for_ids(
                task_split, task_ids, include_samples, include_ground_truth_tasks
            )
        ]

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
        curr_iteration = 0  # TODO: implement resume.
        max_iterations = config[EXPERIMENT_ITERATOR][MAX_ITERATIONS]

        experiment_state.curr_iteration = curr_iteration

        task_batcher = task_loaders.TaskBatcherRegistry.get(
            config[EXPERIMENT_ITERATOR][TASK_BATCHER],
            experiment_state=experiment_state,
            curr_iteration=curr_iteration,
            max_iterations=max_iterations,
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

        print(
            f"task_ids: {len(task_batch_ids)} tasks: {task_batch_ids[0]} -- {task_batch_ids[-1]}"
        )
        print(f"====================================")

    def execute_model_fn(self, experiment_state, curr_loop_block):
        """Executes a model function on a batch of tasks. Model functions should be of the form:

        model_fn(experiment_state, task_split, task_batch_ids, **params...)
        """
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
        experiment_state.checkpoint_models(curr_loop_block[MODELS_TO_CHECKPOOINT])
