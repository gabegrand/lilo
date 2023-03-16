"""
stitch_rewriter.py | Author: Catherine Wong.
Program rewriter model that uses the Stitch compressor to rewrite programs wrt. a grammar.

Updates FRONTIERS given a grammar.
"""

import json
import os
from collections import defaultdict

import stitch_core as stitch

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import EtaExpandFailure, EtaLongVisitor, Program
from src.models.stitch_base import StitchBase

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.PROGRAM_REWRITER]


@ModelRegistry.register
class StitchProgramRewriter(StitchBase, model_loaders.ModelLoader):
    name = "stitch_rewriter"

    # Inventions from prior run of Stitch to use in rewriting process
    compress_output_filename = "stitch_compress_output.json"

    # Programs for Stitch to rewrite
    rewrite_input_filename = "stitch_rewrite_input.json"

    # Output of rewriter
    rewrite_output_filename = "stitch_rewrite_output.json"

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
        beta_reduce_programs: bool = True,
        load_inventions_from_split: str = "train",
        compute_likelihoods: bool = False,
    ):
        """
        Updates experiment_state frontiers wrt. the experiment_state.models[GRAMMAR]

        params:
            `beta_reduce_programs`: Whether to beta reduce programs before compression.
                This will rewrite the programs into the base DSL, removing any abstractions.
            `load_inventions_from_split`: Name of split associated with Stitch inventions.
                Allows rewriting to be performed w/r/t inventions from any split.
            `compute_likelihoods`: Whether to compute log likelihoods of each program
                under the grammar. This requires converting the programs to eta-long form,
                which is error-prone, so we don't do it by default.

        """
        abstractions_filepath = self._get_filepath_for_current_iteration(
            experiment_state.get_checkpoint_directory(),
            StitchProgramRewriter.compress_output_filename,
            split=load_inventions_from_split,
        )
        if not os.path.exists(abstractions_filepath):
            raise FileNotFoundError(abstractions_filepath)
        with open(abstractions_filepath, "r") as f:
            abstractions_data = json.load(f)
        abstractions = [
            stitch.Abstraction(name=abs["name"], body=abs["body"], arity=abs["arity"])
            for abs in abstractions_data["abstractions"]
        ]

        for split in task_splits:
            programs_filepath = self._get_filepath_for_current_iteration(
                experiment_state.get_checkpoint_directory(),
                StitchProgramRewriter.rewrite_input_filename,
                split=split,
            )
            out_filepath = self._get_filepath_for_current_iteration(
                experiment_state.get_checkpoint_directory(),
                StitchProgramRewriter.rewrite_output_filename,
                split=split,
            )
            self.write_frontiers_to_file(
                experiment_state,
                task_splits=[split],
                task_ids_in_splits=task_ids_in_splits,
                frontiers_filepath=programs_filepath,
                beta_reduce_programs=beta_reduce_programs,
                include_samples=include_samples,
            )
            with open(programs_filepath, "r") as f:
                frontiers_dict = json.load(f)
                stitch_kwargs = stitch.from_dreamcoder(frontiers_dict)

            stitch_kwargs.update(dict(eta_long=True, utility_by_rewrite=True))

            rewrite_result = stitch.rewrite(
                programs=stitch_kwargs["programs"],
                abstractions=abstractions,
            )

            programs_rewritten = rewrite_result.json["rewritten_dreamcoder"]
            assert len(programs_rewritten) == len(stitch_kwargs["programs"])
            assert len(programs_rewritten) == len(stitch_kwargs["tasks"])

            task_to_programs = defaultdict(list)
            for task, program in zip(stitch_kwargs["tasks"], programs_rewritten):
                task_to_programs[task].append(program)

            with open(out_filepath, "w") as f:
                json.dump(rewrite_result.json, f, indent=4)

            for task in experiment_state.get_tasks_for_ids(
                task_split=split, task_ids=task_ids_in_splits[split]
            ):
                # If we don't have any solved programs for this task, skip it
                if task.name not in task_to_programs:
                    continue

                frontier_rewritten = Frontier(
                    frontier=[],
                    task=task,
                )
                for p_str in task_to_programs[task.name]:
                    p = Program.parse(p_str)
                    # Catch fatal error when computing likelihood summaries
                    if compute_likelihoods:
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
                if compute_likelihoods:
                    frontier_rewritten = experiment_state.models[
                        model_loaders.GRAMMAR
                    ].rescoreFrontier(frontier_rewritten)

                experiment_state.task_frontiers[split][task] = frontier_rewritten
