"""
stitch_proposer.py | Author : Catherine Wong.

Library learning model that uses the Stitch compressor to propose libraries.
Expects an experiment_state with a GRAMMAR and FRONTIERs.
Updates GRAMMAR based on Stitch compression.
"""
import json
import os
import subprocess

import src.models.model_loaders as model_loaders
from dreamcoder.program import Program

LibraryLearnerRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LIBRARY_LEARNER
]


@LibraryLearnerRegistry.register
class StitchProposerLibraryLearner(model_loaders.ModelLoader):

    name = "stitch_proposer"

    stitch_input_frontiers_file = "stitch_frontiers.json"
    stitch_output_file = "stitch_output.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return StitchProposerLibraryLearner(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def get_compressed_grammar_mdl_prior_rank(
        self, experiment_state, task_splits, task_ids_in_splits, **kwargs
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) based on the training data description length to rerank the libraries.
        """
        # Write frontiers for stitch.
        input_frontiers_file = self._write_frontiers_for_stitch(
            experiment_state, task_splits, task_ids_in_splits
        )

        # Call stitch compressor.
        inventions_list = self._get_stitch_libraries(
            experiment_state,
            input_frontiers_file,
            max_arity=kwargs["max_arity"],
            iterations=kwargs["iterations"],
            candidates_per_iteration=kwargs["candidates_per_iteration"],
        )

        # Update experiment_state grammar.
        new_inventions = [(0., Program.parse(inv)) for inv in inventions_list]
        experiment_state.models[model_loaders.GRAMMAR].productions.extend(new_inventions)

    def get_compressed_grammar_lm_prior_rank(
        self, experiment_state, task_splits, task_ids_in_splits, max_arity, iterations
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(library) under a language model (default Codex) to rerank the libraries.
        """
        # grammar:
        # experiment_state.task_frontiers

    def get_compressed_grammar_lm_alignment_rank(
        self, experiment_state, task_splits, task_ids_in_splits, max_arity, iterations
    ):
        """
        Updates experiment_state.models[GRAMMAR].
        Uses Stitch compressor to propose libraries.
        Uses p(language, libraries) under a language model (default Codex) to rerank the libraries.
        """
        # catwong: here's how you could get an example grammar.
        # catwong: here's how you get all of the language out of the

    def _write_frontiers_for_stitch(
        self, experiment_state, task_splits, task_ids_in_splits
    ):
        """
        Writes out frontiers for Stitch compressor.
        Returns filepath for calling stitch compressor.
        """
        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=False,
        )
        programs = []
        for split in frontiers:
            for frontier in frontiers[split]:
                frontier_programs = [
                    str(entry.program).replace("lambda", "lam") for entry in frontier
                ]
                programs += frontier_programs
        # Write out the programs.
        frontier_file = os.path.join(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.stitch_input_frontiers_file,
        )
        with open(frontier_file, "w") as f:
            json.dump(programs, f)
        return frontier_file

    def _get_stitch_libraries(
        self,
        experiment_state,
        input_frontiers_file,
        max_arity,
        iterations,
        candidates_per_iteration,
    ):
        output_file = os.path.join(
            experiment_state.get_checkpoint_directory(),
            StitchProposerLibraryLearner.stitch_output_file,
        )
        stitch_arguments = {
            "file": os.path.join(os.getcwd(), input_frontiers_file),
            "out": os.path.join(os.getcwd(), output_file),
            "max-arity": max_arity,
            "iterations": iterations,
        }
        stitch_base_command = "cd stitch; cargo run --bin=compress --release -- "
        stitch_command = stitch_base_command + " ".join(
            [f"--{k}={v}" for k, v in stitch_arguments.items()]
        )
        print("Running Stitch with the following command:")
        print(stitch_command)

        subprocess.run(stitch_command, capture_output=True, check=True, shell=True)

        with open(output_file, "r") as f:
            stitch_results = json.load(f)

        inventions_list = [inv["body"] for inv in stitch_results["invs"]]
        return inventions_list