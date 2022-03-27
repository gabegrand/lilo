"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""

import json

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import EtaExpandFailure, EtaLongVisitor, Program
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
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        include_samples,
        load_inventions_from_split: str = "train",
    ):
        """
        Updates experiment_state frontiers wrt. the experiment_state.models[GRAMMAR]

        params:
            `load_inventions_from_split`: Name of split associated with Stitch inventions.
                Allows rewriting to be performed w/r/t inventions from any split.

        """
        inventions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.inventions_filename,
            split=load_inventions_from_split,
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
                include_samples=include_samples,
            )
            self.run_binary(
                bin="rewrite",
                stitch_args=["--dreamcoder-output"],
                stitch_kwargs={
                    "program-file": programs_filepath,
                    "inventions-file": inventions_filepath,
                    "out": out_filepath,
                },
            )

            with open(out_filepath, "r") as f:
                data = json.load(f)
                task_to_programs = {d["task"]: d["programs"] for d in data["frontiers"]}

            for task in experiment_state.get_tasks_for_ids(
                task_split=split, task_ids=task_ids_in_splits[split]
            ):
                frontier_rewritten = Frontier(
                    frontier=[],
                    task=task,
                )
                for program_data in task_to_programs[task.name]:
                    p_str = program_data["program"]
                    p = Program.parse(p_str)
                    # Hack to avoid fatal error when computing likelihood summaries
                    try:
                        p = EtaLongVisitor(request=task.request).execute(p)
                    except EtaExpandFailure:
                        raise EtaExpandFailure(p_str)
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
