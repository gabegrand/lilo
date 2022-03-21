"""
stitch_base.py | Author: Gabe Grand.
Base class containing utilities for working with the Stitch library.

https://github.com/mlb2251/stitch
"""

import json
import os
import subprocess


class StitchBase(object):
    def run_binary(
        self,
        bin: str = "compress",
        stitch_args: list = [],
        stitch_kwargs: dict = {},
    ):
        """Calls `cargo run` to invoke Stitch via subprocess call.

        params:
            bin: Stitch binary.
            stitch_args: Positional arguments to Stitch CLI.
            stitch_kwargs: Keyword arguments to pass to Stitch CLI.

        """
        assert stitch_args or stitch_kwargs
        if "fmt" not in stitch_kwargs:
            stitch_kwargs["fmt"] = "dreamcoder"
        stitch_command = (
            f"cd stitch; cargo run --bin={bin} --release -- {' '.join(stitch_args)} "
        )
        stitch_command += " ".join([f"--{k}={v}" for k, v in stitch_kwargs.items()])
        print("Running Stitch with the following command:")
        print(stitch_command)

        subprocess.run(stitch_command, capture_output=True, check=True, shell=True)

    def write_frontiers_to_file(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        frontiers_filepath: str,
        include_samples: bool = False,
    ):
        """Dumps programs from frontiers to a file that can be passed to Stitch.

        returns:
            Path to JSON file containing a list of programs.
        """
        if len(task_splits) != 1:
            raise ValueError(
                "`write_frontiers_to_file` supports only a single split at a time"
            )

        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=include_samples,
        )
        frontiers_list = []
        for frontier in frontiers[task_splits[0]]:
            frontiers_list.append(
                {
                    "task": frontier.task.name,
                    "programs": [{"program": str(entry.program)} for entry in frontier],
                }
            )

        # Write out the programs.
        os.makedirs(os.path.dirname(frontiers_filepath), exist_ok=True)
        with open(frontiers_filepath, "w") as f:
            json.dump(
                {
                    "DSL": experiment_state.models["grammar"].json(),
                    "frontiers": frontiers_list,
                },
                f,
            )

    def get_inventions_from_file(self, stitch_output_filepath: str):
        with open(stitch_output_filepath, "r") as f:
            stitch_results = json.load(f)

        inv_name_to_dc_fmt = {
            inv["name"]: inv["dreamcoder"] for inv in stitch_results["invs"]
        }

        # Replace `inv0` with inlined definitions in dreamcoder format
        for inv_name, inv_dc_fmt in inv_name_to_dc_fmt.items():
            for prior_inv_name, prior_inv_dc_fmt in inv_name_to_dc_fmt.items():
                # Assume ordered dict with inventions inv0, inv1, ...
                # inv_i only includes prior inventions inv0, ..., inv_i-1
                if prior_inv_name == inv_name:
                    break
                inv_dc_fmt.replace(prior_inv_name, prior_inv_dc_fmt)
            inv_name_to_dc_fmt[inv_name] = inv_dc_fmt

        return inv_name_to_dc_fmt

    def get_inventions_and_metadata_from_file(self, stitch_output_filepath: str):
        with open(stitch_output_filepath, "r") as f:
            stitch_results = json.load(f)

        invs_and_metadata = {
            inv["name"]: {
                "name": inv["name"],
                "body": inv["body"],
                "dreamcoder": inv["dreamcoder"],
                "rewritten": inv["rewritten"],
            }
            for inv in stitch_results["invs"]
        }
        return invs_and_metadata

    def _get_filepath_for_current_iteration(
        self,
        checkpoint_directory: str,
        filename: str,
        split: str = "",
    ):
        return os.path.join(
            os.getcwd(),
            checkpoint_directory,
            split,
            filename,
        )
