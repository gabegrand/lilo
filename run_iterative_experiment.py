import json
import os

from run_experiment import *
from src.experiment_iterator import EXPORT_DIRECTORY


def main(args):
    for global_batch_size in [10, 20, 30]:
        config = load_config_from_file(args)
        config["experiment_iterator"]["task_batcher"]["params"][
            "global_batch_size"
        ] = global_batch_size
        config["metadata"]["experiment_id"] += f"_{global_batch_size}"

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
