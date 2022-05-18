"""
run_iterative_experiment.py | Author: Gabe Grand.

Like run_experiment.py, but runs multiple experiments with different global
batch sizes.

Writes results to experiments_iterative directory.

Usage:

python run_iterative_experiment.py \
	--experiment_type stitch \
	--domain logo \
	--stitch_params '{"iterations": 10}'

python run_iterative_experiment.py \
	--experiment_type stitch_codex \
	--domain logo \
	--stitch_params '{"iterations": 10}' \
	--codex_params '{"use_cached": true}'

python run_iterative_experiment.py \
	--experiment_type oracle \
	--domain logo \
	--stitch_params '{"iterations": 10}'

By default, runs a single replication of the experiment. To run multiple
replications with different random seeds, use the `--random_seeds` flag.

"""

import argparse
import json
import os

from run_experiment import init_experiment_state_and_iterator, run_experiment
from src.config_builder import build_config
from src.experiment_iterator import EXPORT_DIRECTORY
from src.task_loaders import GroundTruthOrderedTaskBatcher

parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment_name", required=True, help="Unique name for this experiment."
)

parser.add_argument(
    "--experiment_type", required=True, help="[stitch, stitch_codex, oracle]"
)

parser.add_argument("--domain", required=True, help="[logo, clevr, re2]")

parser.add_argument(
    "--task_batcher",
    default=GroundTruthOrderedTaskBatcher.name,
    help="[ground_truth_ordered_task_batcher, random_shuffle_ordered_task_batcher]",
)

parser.add_argument(
    "--increment_task_batcher",
    default=False,
    action="store_true",
    help="Increment the task batch pointer over the global ordering at each iteration. By default, we turn this off - each iteration uses a fixed initial set of tasks. Turning this on allows a data-loader style sliding window over the training set, as in iterative search and synthesis in DC: you get a fresh batch of tasks at each outer loop of a single experiment.",
)

parser.add_argument(
    "--iterations",
    default=1,
    type=int,
    help="How many iterations to run the experiment loop specified in the config.",
)

parser.add_argument(
    "--stitch_params", default="{}", help="JSON string of stitch params"
)

parser.add_argument("--codex_params", default="{}", help="JSON string of codex params")

parser.add_argument(
    "--compute_likelihoods",
    action="store_true",
    help="Compute program log likelihoods",
)

parser.add_argument(
    "--global_batch_sizes",
    nargs="+",
    default=[],
    type=int,
    help="List of global_batch_size values, one per iteration.",
)

parser.add_argument(
    "--random_seeds",
    nargs="+",
    default=[0],
    type=int,
    help="List of random seed values. Each one generates a full replication of the experiment.",
)

parser.add_argument(
    "--use_cached",
    default=False,
    action="store_true",
    help="Use cached versions of Codex queries.",
)


def main(args):

    codex_params = json.loads(args.codex_params)
    stitch_params = json.loads(args.stitch_params)
    if args.use_cached:
        codex_params["use_cached"] = True

    for random_seed in args.random_seeds:
        config_base = build_config(
            experiment_name=args.experiment_name,
            experiment_type=args.experiment_type,
            domain=args.domain,
            task_batcher=args.task_batcher,
            random_seed=random_seed,
            iterations=args.iterations,
            codex_params=codex_params,
            stitch_params=stitch_params,
            compute_likelihoods=args.compute_likelihoods,
            compute_description_lengths=True,
            increment_task_batcher=args.increment_task_batcher,
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

        for global_batch_size in args.global_batch_sizes:
            config = build_config(
                experiment_name=args.experiment_name,
                experiment_type=args.experiment_type,
                domain=args.domain,
                task_batcher=args.task_batcher,
                random_seed=random_seed,
                iterations=args.iterations,
                global_batch_size=global_batch_size,
                codex_params=codex_params,
                stitch_params=stitch_params,
                compute_likelihoods=args.compute_likelihoods,
                compute_description_lengths=True,
                increment_task_batcher=args.increment_task_batcher,
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
