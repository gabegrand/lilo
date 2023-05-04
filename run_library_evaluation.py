"""
run_library_evaluation.py | Author: Gabe Grand.

Evaluate and compare unconditional synthesis performance of the final library across
multiple experiment conditions.

"""

import argparse
import json
import os
import shutil

from run_experiment import init_experiment_state_and_iterator, run_experiment
from src.config_builder import build_config
from src.experiment_iterator import EXPORT_DIRECTORY
from src.logging_utils import OutputLogger
from src.task_loaders import ALL

parser = argparse.ArgumentParser()

parser.add_argument("--domain", required=True, help="[logo, clevr, re2]")

parser.add_argument(
    "--prior_experiment_name", required=True, help="Prior experiment_name to load."
)

parser.add_argument(
    "--prior_experiment_types",
    nargs="+",
    type=str,
    required=True,
    help="[stitch, stitch_codex, oracle]",
)

parser.add_argument(
    "--prior_batch_size",
    required=True,
    type=int,
    help="Batch size of prior runs. Must be the same across all runs.",
)

parser.add_argument(
    "--final_iteration",
    required=True,
    type=int,
    help="Initialize from a later iteration; e.g., init_iteration=3 starts the experiment at iteration 3.",
)

parser.add_argument(
    "--random_seeds",
    nargs="+",
    default=[0],
    type=int,
    help="List of random seed values. Each one generates a full replication of the experiment.",
)

parser.add_argument(
    "--enumeration_timeout",
    default=None,
    type=int,
    help="How many seconds to run enumeration for. Applies to both train and test sets. Defaults to whatever value is specified in the experiment template.",
)

parser.add_argument(
    "--overwrite_dir",
    default=False,
    action="store_true",
    help="Overwrites any existing files associated with `experiment_name` in export and log directories.",
)

parser.add_argument(
    "--no_s3_sync",
    default=False,
    action="store_true",
    help="Disable AWS S3 upload.",
)

EXPERIMENT_TYPE_LIBRARY_EVALUATION = "library_evaluation"


def main(args):

    experiment_name = f"library_evaluation_{args.prior_experiment_name}"

    for random_seed in args.random_seeds:
        for experiment_type in args.prior_experiment_types:

            resume_checkpoint_directory = os.path.join(
                "experiments_iterative",
                "outputs",
                args.prior_experiment_name,
                "domains",
                args.domain,
                experiment_type,
                f"seed_{random_seed}",
                f"{experiment_type}_{args.prior_batch_size}",
            )
            if not os.path.exists(resume_checkpoint_directory):
                raise ValueError(resume_checkpoint_directory)

            export_directory = os.path.join(
                "experiments_iterative",
                "outputs",
                experiment_name,
                "domains",
                args.domain,
                experiment_type,
                f"seed_{random_seed}",
            )

            config_base = build_config(
                experiment_name=experiment_name,
                experiment_type=EXPERIMENT_TYPE_LIBRARY_EVALUATION,
                custom_experiment_type=experiment_type,
                domain=args.domain,
                random_seed=random_seed,
                iterations=args.final_iteration + 1,
                init_iteration=args.final_iteration,
                enumeration_timeout=args.enumeration_timeout,
                gpt_params={},
                stitch_params={},
                compute_likelihoods=True,
                compute_description_lengths=True,
                increment_task_batcher=True,
                init_grammar_from_checkpoint=True,
                resume_checkpoint_directory=resume_checkpoint_directory,
                s3_sync=(not args.no_s3_sync),
            )
            global_batch_sizes = [ALL]
            config_base["metadata"]["global_batch_sizes"] = global_batch_sizes

            # Delete any existing files associated with this experiment name.
            if args.overwrite_dir:
                export_dir = config_base["metadata"]["export_directory"]
                if os.path.exists(export_dir):
                    shutil.rmtree(export_dir)
                log_dir = config_base["metadata"]["log_directory"]
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir)

            # Write a copy of config.json to the experiment directory
            config_base_write_path = os.path.join(export_directory, "config_base.json")
            os.makedirs(os.path.dirname(config_base_write_path), exist_ok=True)
            with open(config_base_write_path, "w") as f:
                json.dump(config_base, f, indent=4)

            experiment_state, experiment_iterator = init_experiment_state_and_iterator(
                args, config_base
            )

            # Write a copy of config.json to the experiment directory
            config_write_path = os.path.join(
                experiment_state.metadata[EXPORT_DIRECTORY], "config.json"
            )
            with open(config_write_path, "w") as f:
                json.dump(config_base, f, indent=4)

            print(f"Evaluating: {resume_checkpoint_directory}")

            log_path = os.path.join(
                config_base["metadata"]["export_directory"], "run.log"
            )
            with OutputLogger(log_path=log_path) as logger:
                try:
                    run_experiment(args, experiment_state, experiment_iterator)
                except:
                    logger.exception(
                        f"Exception encountered while running experiment. See logs at: {log_path}"
                    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
