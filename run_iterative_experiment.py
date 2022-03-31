"""
run_iterative_experiment.py | Author: Gabe Grand.

Like run_experiment.py, but runs multiple experiments with different global
batch sizes.

Each iteration of the experiment is written to a subdirectory:

`{export_directory}/experiment_id/experiment_id_{batch_size}/`

Usage:

python run_iterative_experiment.py \
    --experiment_type stitch \
    --domain logo
    --global_batch_sizes 5 10 15 25 50 100 150 200

"""

import argparse
import json
import os
import shutil

from run_experiment import init_experiment_state_and_iterator, run_experiment
from src.config_builder import build_config
from src.experiment_iterator import EXPORT_DIRECTORY

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_type", required=True)

parser.add_argument("--domain", required=True)

parser.add_argument("--stitch_params", default="{}")

parser.add_argument("--codex_params", default="{}")

parser.add_argument(
    "--no_likelihoods",
    action="store_true",
    help="Do not compute likelihoods",
)

parser.add_argument(
    "--global_batch_sizes",
    nargs="+",
    default=[],
    type=int,
    help="List of global_batch_size values, one per iteration.",
)

parser.add_argument(
    "--overwrite",
    default=False,
    action="store_true",
    help="Delete the `experiment_id` directory before running the experiment.",
)


def main(args):

    config_base = build_config(
        experiment_type=args.experiment_type,
        domain=args.domain,
        codex_params=json.loads(args.codex_params),
        stitch_params=json.loads(args.stitch_params),
        compute_likelihoods=(not args.no_likelihoods),
        compute_description_lengths=True,
    )

    # If --global_batch_sizes is not specified, use the domain-specific default.
    if not args.global_batch_sizes:
        args.global_batch_sizes = config_base["metadata"]["global_batch_sizes"]
    config_base["metadata"]["global_batch_sizes"] = args.global_batch_sizes

    # Write a copy of config.json to the experiment directory
    config_base_write_path = os.path.join(
        config_base["metadata"]["export_directory"], "config_base.json"
    )
    os.makedirs(os.path.dirname(config_base_write_path), exist_ok=True)
    with open(config_base_write_path, "w") as f:
        json.dump(config_base, f, indent=4)

    # Clear the experiment_id_base directory
    if args.overwrite:
        shutil.rmtree(
            os.path.join(
                os.getcwd(),
                config_base["metadata"]["export_directory"],
            ),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(
                os.getcwd(),
                config_base["metadata"]["log_directory"],
            ),
            ignore_errors=True,
        )

    for global_batch_size in args.global_batch_sizes:
        config = build_config(
            experiment_type=args.experiment_type,
            domain=args.domain,
            global_batch_size=global_batch_size,
            codex_params=json.loads(args.codex_params),
            stitch_params=json.loads(args.stitch_params),
            compute_likelihoods=(not args.no_likelihoods),
            compute_description_lengths=True,
        )

        experiment_state, experiment_iterator = init_experiment_state_and_iterator(
            args, config
        )

        # Write a copy of config.json to the experiment directory
        config_write_path = os.path.join(
            experiment_state.metadata[EXPORT_DIRECTORY], "config.json"
        )
        with open(config_write_path, "w") as f:
            json.dump(config, f, indent=4)

        run_experiment(args, experiment_state, experiment_iterator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
