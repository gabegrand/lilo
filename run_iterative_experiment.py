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
	--gpt_params '{"use_cached": true}'

python run_iterative_experiment.py \
	--experiment_type oracle \
	--domain logo \
	--stitch_params '{"iterations": 10}'

python run_iterative_experiment.py \
	--experiment_type enumeration \
	--domain re2

By default, runs a single replication of the experiment. To run multiple
replications with different random seeds, use the `--random_seeds` flag.

"""

import argparse
import json
import os
import shutil

from run_experiment import init_experiment_state_and_iterator, run_experiment
from src.config_builder import build_config, get_domain_metadata
from src.experiment_iterator import EXPORT_DIRECTORY
from src.logging_utils import OutputLogger
from src.task_loaders import ALL, RandomShuffleOrderedTaskBatcher
from src.utils import write_command_to_file

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
    default=RandomShuffleOrderedTaskBatcher.name,
    help="[ground_truth_ordered_task_batcher, random_shuffle_ordered_task_batcher]",
)

parser.add_argument(
    "--iterations",
    default=1,
    type=int,
    help="How many LAPS iterations to run the experiment loop specified in the config.",
)

parser.add_argument(
    "--enumeration_timeout",
    default=None,
    type=int,
    help="How many seconds to run enumeration for. Applies to both train and test sets. Defaults to whatever value is specified in the experiment template.",
)

parser.add_argument(
    "--recognition_train_steps",
    default=None,
    type=int,
    help="How many training steps to train the recognition model for. Defaults to whatever value is specified in the experiment template.",
)

parser.add_argument(
    "--encoder",
    default=None,
    type=str,
    help="The name of the encoder to use in the recognition model. Defaults to whatever value is specified in the experiment template.",
)

parser.add_argument(
    "--stitch_params", default="{}", help="JSON string of Stitch params"
)

parser.add_argument("--gpt_params", default="{}", help="JSON string of GPT params")

parser.add_argument(
    "--no_likelihoods",
    action="store_true",
    help="[DEPRAECATED: Eta long issues should be fixed now] Disable computing program log likelihoods, which sometimes produce EtaExpandFailure errors",
)

parser.add_argument(
    "--global_batch_sizes",
    nargs="+",
    default=[],
    type=int,
    help="List of global_batch_size values, one per iteration.",
)

parser.add_argument(
    "--global_batch_size_all",
    default=False,
    action="store_true",
    help="Set global_batch_size to all",
)

parser.add_argument(
    "--random_seeds",
    nargs="+",
    default=[0],
    type=int,
    help="List of random seed values. Each one generates a full replication of the experiment.",
)

parser.add_argument(
    "--init_iteration",
    default=0,
    type=int,
    help="Initialize from a later iteration; e.g., init_iteration=3 starts the experiment at iteration 3.",
)

parser.add_argument(
    "--init_frontiers_from_checkpoint",
    default=False,
    action="store_true",
    help="Initialize the frontiers from a checkpoint (location automatically inferred).",
)

parser.add_argument(
    "--init_frontiers_every_iteration",
    default=False,
    action="store_true",
    help="If using init_frontiers_from_checkpoint, default is to init once at the start. With --init_frontiers_every_iteration, frontiers are also loaded at every subsequent iteration.",
)

parser.add_argument(
    "--init_grammar_from_checkpoint",
    default=False,
    action="store_true",
    help="Initialize the grammar from a checkpoint (location automatically inferred). Occurs by default when --init_frontiers_from_checkpoint=True",
)

parser.add_argument(
    "--resume_checkpoint_directory",
    default=None,
    type=str,
    help="If using init_frontiers_from_checkpoint, optionally point to a checkpoint from a different experiment run.",
)

parser.add_argument(
    "--use_cached",
    default=False,
    action="store_true",
    help="Use cached versions of GPT queries.",
)

parser.add_argument(
    "--resume_strategy",
    default=None,
    choices=["first", "every"],
    help="First: Resume first iteration only. Every: Resume every iteration.",
)

parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="Replaces live query to GPT with a random sample from the training set.",
)

parser.add_argument(
    "--verbose",
    default=False,
    action="store_true",
    help="Enables additional prints for certain models.",
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

parser.add_argument(
    "--embedding",
    default=False,
    action="store_true",
    help="using embedding to select examples",
)


def main(args):

    gpt_params = json.loads(args.gpt_params)
    stitch_params = json.loads(args.stitch_params)
    if args.resume_strategy:
        gpt_params["resume_strategy"] = args.resume_strategy
    if args.use_cached:
        gpt_params["use_cached"] = True
    if args.debug:
        gpt_params["debug"] = True
    if args.verbose:
        gpt_params["verbose"] = True

    for random_seed in args.random_seeds:
        config_base = build_config(
            experiment_name=args.experiment_name,
            experiment_type=args.experiment_type,
            domain=args.domain,
            task_batcher=args.task_batcher,
            random_seed=random_seed,
            iterations=args.iterations,
            init_iteration=args.init_iteration,
            enumeration_timeout=args.enumeration_timeout,
            recognition_train_steps=args.recognition_train_steps,
            encoder=args.encoder,
            gpt_params=gpt_params,
            stitch_params=stitch_params,
            compute_likelihoods=(not args.no_likelihoods),
            compute_description_lengths=True,
            increment_task_batcher=True,
            init_frontiers_from_checkpoint=args.init_frontiers_from_checkpoint,
            init_frontiers_every_iteration=args.init_frontiers_every_iteration,
            init_grammar_from_checkpoint=args.init_grammar_from_checkpoint,
            resume_checkpoint_directory=args.resume_checkpoint_directory,
            s3_sync=(not args.no_s3_sync),
            embedding=args.embedding,
        )

        if args.global_batch_size_all:
            # Runs a single iteration with all tasks.
            global_batch_sizes = [ALL]
        elif args.global_batch_sizes:
            # Runs multiple iterations with a manually-specified list of batch sizes.
            global_batch_sizes = args.global_batch_sizes
        else:
            # If --global_batch_sizes is not specified, use the domain-specific default.
            global_batch_sizes = get_domain_metadata(args.domain)["global_batch_sizes"]
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
        config_base_write_path = os.path.join(
            config_base["metadata"]["export_directory"], "config_base.json"
        )
        os.makedirs(os.path.dirname(config_base_write_path), exist_ok=True)
        with open(config_base_write_path, "w") as f:
            json.dump(config_base, f, indent=4)

        write_command_to_file(
            source_python_file=os.path.basename(__file__),
            args=args,
            save_path=os.path.join(
                config_base["metadata"]["export_directory"], "run_args.txt"
            ),
        )

        for global_batch_size in global_batch_sizes:
            config = build_config(
                experiment_name=args.experiment_name,
                experiment_type=args.experiment_type,
                domain=args.domain,
                task_batcher=args.task_batcher,
                random_seed=random_seed,
                init_iteration=args.init_iteration,
                iterations=args.iterations,
                enumeration_timeout=args.enumeration_timeout,
                recognition_train_steps=args.recognition_train_steps,
                encoder=args.encoder,
                global_batch_size=global_batch_size,
                gpt_params=gpt_params,
                stitch_params=stitch_params,
                compute_likelihoods=(not args.no_likelihoods),
                compute_description_lengths=True,
                increment_task_batcher=True,
                init_frontiers_from_checkpoint=args.init_frontiers_from_checkpoint,
                init_frontiers_every_iteration=args.init_frontiers_every_iteration,
                init_grammar_from_checkpoint=args.init_grammar_from_checkpoint,
                resume_checkpoint_directory=args.resume_checkpoint_directory,
                s3_sync=(not args.no_s3_sync),
                embedding=args.embedding,
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

            log_path = os.path.join(
                experiment_state.metadata[EXPORT_DIRECTORY], "run.log"
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
