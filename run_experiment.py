"""
run_experiment.py | Author: Catherine Wong.

Commmand line utility for running experiments.
Requires a CONFIG file specifying the experiment.

By default, it exports:
    Experiment outputs and results to the export_directory.
    Experiment logs to the log_directory.

Usage:
    python run_experiment.py
        --config_dir experiments/configs
        --config_file dreamcoder_compositional_graphics_200_human.json
"""
import argparse
import json
import os

from src.experiment_iterator import ExperimentIterator, ExperimentState

"""zyzzyva@ - Temporarily disable the LAX domains. These incorrectly overwrite the global int primitives to floats, breaking
backward compatability with the DC domains that rely on integers.

from data.drawings.make_tasks import *
from data.drawings.grammar import *

from data.structures.make_tasks import *
from data.structures.grammar import *
"""


DEFAULT_CONFIG_DIR = "experiments/configs"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_dir",
    default=DEFAULT_CONFIG_DIR,
    help="Top level directory containing experiment config files.",
)
parser.add_argument(
    "--config_file",
    required=True,
    help="File name of the config within the config directory.",
)


def load_config_from_file(args):
    config_full_path = os.path.join(args.config_dir, args.config_file)
    with open(config_full_path) as f:
        return json.load(f)


def init_experiment_state_and_iterator(args, config):
    experiment_state = ExperimentState(config)
    experiment_iterator = ExperimentIterator(config, experiment_state)
    return experiment_state, experiment_iterator


def run_experiment(args, experiment_state, experiment_iterator):
    while not experiment_iterator.is_finished():
        experiment_iterator.next(experiment_state)


def main(args):
    config = load_config_from_file(args)
    experiment_state, experiment_iterator = init_experiment_state_and_iterator(
        args, config
    )
    run_experiment(args, experiment_state, experiment_iterator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
