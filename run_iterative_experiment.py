import json
import os
import shutil

from run_experiment import *
from src.experiment_iterator import EXPORT_DIRECTORY
from src.models.model_loaders import SAMPLE_GENERATOR

# GLOBAL_BATCH_SIZES = [5, 15, 25, 50, 100, 150, 200]
parser.add_argument(
    "--global_batch_sizes",
    nargs="+",
    default=[5, 10, 15],
    type=int,
)

parser.add_argument(
    "--use_cached",
    default=False,
    action="store_true",
)

parser.add_argument(
    "--overwrite",
    default=False,
    action="store_true",
)


def main(args):
    config = load_config_from_file(args)

    # Clear the experiment_id_base directory
    if args.overwrite:
        shutil.rmtree(
            os.path.join(
                os.getcwd(),
                config["metadata"]["export_directory"],
                config["metadata"]["experiment_id"],
            ),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(
                os.getcwd(),
                config["metadata"]["log_directory"],
                config["metadata"]["experiment_id"],
            ),
            ignore_errors=True,
        )

    for global_batch_size in args.global_batch_sizes:
        config = load_config_from_file(args)

        # Create a dedicated directory for all iterations of this experiment
        experiment_id_base = config["metadata"]["experiment_id"]
        config["metadata"]["export_directory"] = os.path.join(
            config["metadata"]["export_directory"], experiment_id_base
        )
        config["metadata"]["log_directory"] = os.path.join(
            config["metadata"]["log_directory"], experiment_id_base
        )
        config["metadata"][
            "experiment_id"
        ] = f"{experiment_id_base}_{global_batch_size}"

        # Update the batch_size
        config["experiment_iterator"]["task_batcher"]["params"][
            "global_batch_size"
        ] = global_batch_size

        # Update any necessary model params in the loop blocks
        loop_blocks = []
        for block in config["experiment_iterator"]["loop_blocks"]:
            if block.get("model_type") == SAMPLE_GENERATOR:
                block["params"]["use_cached"] = args.use_cached
            loop_blocks.append(block)
        config["loop_blocks"] = loop_blocks

        experiment_state, experiment_iterator = init_experiment_state_and_iterator(
            args, config
        )

        # Write a copy of config.json to the experiment directory
        config_write_path = os.path.join(
            experiment_state.metadata[EXPORT_DIRECTORY], "config.json"
        )
        with open(config_write_path, "w") as f:
            json.dump(config, f)

        run_experiment(args, experiment_state, experiment_iterator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
