"""
stitch_base.py | Author: Gabe Grand.
Base class containing utilities for working with the Stitch library.

https://github.com/mlb2251/stitch
"""

import json
import os
import subprocess

from src.models.model_loaders import GRAMMAR


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
        use_mdl_program: bool = False,
        beta_reduce_programs: bool = False,
        include_samples: bool = True,
    ):
        """Dumps programs from frontiers to a file that can be passed to Stitch.

        returns:
            Path to JSON file containing a list of programs.
        """

        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=include_samples,
        )
        frontiers_list = []
        for split in task_splits:
            for frontier in frontiers[split]:
                programs = [entry.program for entry in frontier]

                if len(programs) > 0:
                    if beta_reduce_programs:
                        programs = [p.betaNormalForm() for p in programs]

                    # TODO: Beta reduce before MDL?
                    if use_mdl_program:
                        programs = [
                            experiment_state.models[GRAMMAR].get_mdl_program(programs)
                        ]

                    frontiers_list.append(
                        {
                            "task": frontier.task.name,
                            "programs": [{"program": str(p)} for p in programs],
                        }
                    )

        # Write out the programs.
        os.makedirs(os.path.dirname(frontiers_filepath), exist_ok=True)
        with open(frontiers_filepath, "w") as f:
            json.dump(
                {
                    "DSL": experiment_state.models[GRAMMAR].json(),
                    "frontiers": frontiers_list,
                },
                f,
                indent=4,
            )

    def get_inventions_from_file(self, stitch_output_filepath: str):
        with open(stitch_output_filepath, "r") as f:
            stitch_results = json.load(f)

        inv_name_to_dc_fmt = {
            inv["name"]: inv["dreamcoder"] for inv in stitch_results["invs"]
        }

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
