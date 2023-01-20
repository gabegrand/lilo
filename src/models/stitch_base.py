"""
stitch_base.py | Author: Gabe Grand.
Base class containing utilities for working with the Stitch library.

https://github.com/mlb2251/stitch
"""

import json
import os

from src.experiment_iterator import RANDOM_GENERATOR
from src.models.model_loaders import GRAMMAR


class StitchBase(object):
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
        rng = experiment_state.metadata[RANDOM_GENERATOR]

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

                    if use_mdl_program:
                        programs = [
                            rng.choice(
                                experiment_state.models[GRAMMAR].get_mdl_programs(
                                    programs
                                )
                            )
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
