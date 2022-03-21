import json
import os

from run_experiment import *
from src.experiment_iterator import EXPORT_DIRECTORY
from src.models.model_loaders import SAMPLE_GENERATOR

parser.add_argument(
    "--use_cached",
    default=False,
)


GLOBAL_BATCH_SIZES = [5, 15, 25, 50, 100, 150, 200]


def main(args):
    for global_batch_size in GLOBAL_BATCH_SIZES:
        config = load_config_from_file(args)
        config["experiment_iterator"]["task_batcher"]["params"][
            "global_batch_size"
        ] = global_batch_size
        config["metadata"]["experiment_id"] += f"_{global_batch_size}"

        loop_blocks = []
        for block in config["experiment_iterator"]["loop_blocks"]:
            if block.get("model_type") == SAMPLE_GENERATOR:
                block["params"]["use_cached"] = args.use_cached
            loop_blocks.append(block)
        config["loop_blocks"] = loop_blocks

        experiment_state, experiment_iterator = init_experiment_state_and_iterator(
            args, config
        )

        config_write_path = os.path.join(
            experiment_state.metadata[EXPORT_DIRECTORY], "config.json"
        )
        with open(config_write_path, "w") as f:
            json.dump(config, f)

        run_experiment(args, experiment_state, experiment_iterator)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
