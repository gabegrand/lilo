"""
run_iterative_experiment.py | Author: Gabe Grand.

Like run_experiment.py, but runs multiple experiments with different global
batch sizes.

Each iteration of the experiment is written to a subdirectory:

`{export_directory}/experiment_id/experiment_id_{batch_size}/`

Usage:

python run_iterative_experiment.py
    --config_file logo_stitch_iterative.json
    --global_batch_sizes 5 10 15 25 50 100 150 200

"""

import argparse
import json
import os
import shutil

from config_builder import (
    build_config,
    init_experiment_state_and_iterator,
    run_experiment,
)
from src.experiment_iterator import EXPORT_DIRECTORY

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_type", required=True)

parser.add_argument("--domain", required=True)

parser.add_argument(
    "--global_batch_sizes",
    nargs="+",
    default=[5, 10, 15],
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

    config = build_config(
        experiment_type=args.experiment_type,
        domain=args.domain,
    )

    # Write a copy of config.json to the experiment directory
    config_write_path = os.path.join(
        config["metadata"]["export_directory"], "config.json"
    )
    with open(config_write_path, "w") as f:
        json.dump(config, f, indent=4)

    # Clear the experiment_id_base directory
    if args.overwrite:
        shutil.rmtree(
            os.path.join(
                os.getcwd(),
                config["metadata"]["export_directory"],
            ),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(
                os.getcwd(),
                config["metadata"]["log_directory"],
            ),
            ignore_errors=True,
        )

    for global_batch_size in args.global_batch_sizes:
        config = build_config(
            experiment_type=args.experiment_type,
            domain=args.domain,
        )

        # Create a dedicated directory for all iterations of this experiment
        config["metadata"][
            "experiment_id"
        ] = f'{config["metadata"]["experiment_id"]}_{global_batch_size}'

        # Update the batch_size
        config["experiment_iterator"]["task_batcher"]["params"][
            "global_batch_size"
        ] = global_batch_size

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
