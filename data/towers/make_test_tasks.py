"""
tower : test_make_tasks.py | Author : Catherine Wong.
"""

from src.task_loaders import *

import data.tower.make_tasks as to_test


def test_load_tower_4_block_unique_silhouettes_load_tasks():
    task_loader = TaskLoaderRegistry[to_test.Tower4BlockUniqueSilhouettesLoader.name]
    tasks = task_loader.load_tasks()
    import pdb

    pdb.set_trace()
