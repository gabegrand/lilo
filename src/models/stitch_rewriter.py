"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""

import json

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import EtaLongVisitor, Program
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
        # There should be a single set of inventions for all splits
        inventions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.inventions_filename,
        )
        for split in task_splits:
            programs_filepath = self._get_filepath_for_current_iteration(
                experiment_state.get_checkpoint_directory(),
                StitchProgramRewriter.programs_filename,
                split=split,
            )
            out_filepath = self._get_filepath_for_current_iteration(
                experiment_state.get_checkpoint_directory(),
                StitchProgramRewriter.out_filename,
                split=split,
            )
            self.write_frontiers_to_file(
                experiment_state,
                task_splits=[split],
                task_ids_in_splits=task_ids_in_splits,
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
                data = json.load(f)
                task_to_programs = {d["task"]: d["programs"] for d in data["frontiers"]}

            # Replace all frontiers for each task with rewritten frontiers
            for task in experiment_state.task_frontiers[split].keys():
                frontier_rewritten = Frontier(
                    frontier=[],
                    task=task,
                )
                for program_data in task_to_programs[task.name]:
                    p_str = self._inline_inventions(
                        program_data["program"], inv_name_to_dc_fmt
                    )
                    p = Program.parse(p_str)
                    # Hack to avoid fatal error when computing likelihood summaries
                    p = EtaLongVisitor(request=task.request).execute(p)
                    frontier_rewritten.entries.append(
                        FrontierEntry(
                            program=p,
                            logPrior=0.0,
                            logLikelihood=0.0,
                        )
                    )
                # Re-score the logPrior and logLikelihood of the frontier under the current grammar
                frontier_rewritten = experiment_state.models[
                    model_loaders.GRAMMAR
                ].rescoreFrontier(frontier_rewritten)

                experiment_state.task_frontiers[split][task] = frontier_rewritten

    def _inline_inventions(self, p_str: str, inv_name_to_dc_fmt: dict):
        # `inv0, inv1, ...` should be in sorted order to avoid partial replacement issues
        assert list(inv_name_to_dc_fmt.keys()) == sorted(
            list(inv_name_to_dc_fmt.keys())
        )
        for inv_name, inv_dc_fmt in inv_name_to_dc_fmt.items():
            p_str = p_str.replace(inv_name, inv_dc_fmt)
        return p_str
