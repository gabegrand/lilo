"""
evaluate_library_learner.py | Author : Catherine Wong.

Library learning and evaluation test harness. Dataset and model should be specified in a config file.
Test harness-specific functions can be expressed in the config or as a hyperparameter.

Usage:
python evaluate_library_learner.py.
"""
import json
import argparse

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


def run_library_evaluation_harness(args):
    # Get the initial experiment state with language and programs.





def main(args):
    config = load_config_from_file(args)
    run_library_evaluation_harness(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
