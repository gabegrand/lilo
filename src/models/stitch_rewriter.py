"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""

import json

import src.models.model_loaders as model_loaders
from src.models.stitch_base import StitchBase

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.PROGRAM_REWRITER]


@ModelRegistry.register
class StitchProgramRewriter(StitchBase, model_loaders.ModelLoader):
    name = "stitch_rewriter"

    # Inventions from prior run of Stitch to use in rewriting process
    inventions_filename = "stitch_output.json"

    # Programs for Stitch to rewrite
    programs_filename = "programs_to_rewrite.json"

    # Output of rewriter
    out_filename = "programs_rewritten.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return StitchProgramRewriter(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def get_rewritten_frontiers_for_grammar(
        self, experiment_state, task_splits, task_ids_in_splits
    ):
        """
        Updates experiment_state frontiers wrt. the experiment_state.models[GRAMMAR]
        """
        inventions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.inventions_filename,
        )
        programs_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.programs_filename,
        )
        out_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.out_filename,
        )
        self.write_frontiers_to_file(
            experiment_state,
            task_splits,
            task_ids_in_splits,
            frontiers_filepath=programs_filepath,
        )
        self.run_binary(
            bin="rewrite",
            stitch_kwargs={
                "program-file": programs_filepath,
                "inventions-file": inventions_filepath,
                "out": out_filepath,
            },
        )

        inv_name_to_dc_fmt = self.get_inventions_from_file(
            stitch_output_filepath=inventions_filepath
        )

        with open(out_filepath, "r") as f:
            rewritten_programs_str = json.load(f)["rewritten"]

        # Replace `inv0, inv1, ...` with inlined inventions
        rewritten_programs_str_inlined = []
        for p_str in rewritten_programs_str:
            p_str = p_str.replace("lam", "lambda")
            for inv_name, inv_dc_fmt in inv_name_to_dc_fmt.items():
                p_str = p_str.replace(inv_name, inv_dc_fmt)
            rewritten_programs_str_inlined.append(p_str)
