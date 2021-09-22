"""
evaluate_compression_model_scoring.py | Author : Catherine Wong.

Evaluation for new models that score candidate library functions. Runs simulated iterations of compression using ground truth programs.

Usage: 
python evaluate_compression_model_scoring.py
    --config_file syntax_robustfill_language_only_compositional_graphics_200_synthetic.json # Config to initialize the model and the dataset. Assumes the amortized synthesis model is the one with the scoring function.
    --library_candidates_scoring_fn # Which scoring function you want to use.
    -k # Pytest keywrods for which test to run (substring of tests), otherwise runs all. 
"""
import os, json, argparse

from src.experiment_iterator import ExperimentState, ExperimentIterator
from data.compositional_graphics.make_tasks import *

# All of the model loaders we import.
from data.compositional_graphics.grammar import *
from data.compositional_graphics.encoder import *

from src.models.laps_dreamcoder_recognition import *

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


def main(args):
    config = load_config_from_file(args)
    experiment_state = ExperimentState(config)
