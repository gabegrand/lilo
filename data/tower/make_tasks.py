"""
towers : make_tasks.py | Author : Catherine Wong.

Utility functions for loading tasks for the towers domain. This domain was designed by McCarthy et. al and builds on the towers domain in DreamCoder.

TODO: under construction!
"""
import os, dill
from src.task_loaders import *

from dreamcoder.domains.tower.makeTowerTasks import SupervisedTower

DEFAULT_DATA_DIRECTORY = "data/tower"


@TaskLoaderRegistry.register
class Tower4BlockUniqueSilhouettesLoader(TaskDataLoader):
    name = "tower_4_block_unique_silhouettes"

    def load_tasks(self):
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, self.name)
        raise NotImplementedError
