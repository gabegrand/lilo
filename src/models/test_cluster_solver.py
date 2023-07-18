import json
import sys
from src.models.sample_generator import GPTSampleGenerator
from src.models.gpt_base import Prompt

path = "experiments_iterative/outputs/runs_gpt_learner/domains/re2/gpt_solver_learner_gpt4/seed_111/gpt_solver_learner_gpt4_96/2/train/gpt_library_abstraction_results.json"
with open(path, "r") as f:
    gpt_abstraction = json.load(f)



def get_abstraction(gpt_abstraction):
    abstractions = gpt_abstraction["abstractions"]
    for abstraction in abstractions:
        return abstraction

def _get_dsl_description(self, include_abstractions: bool = True):
    dsl_fns = []
    for primitive in self.grammar.primitives:
        if primitive.isInvented and (not include_abstractions):
            # Optionally, skip abstractions
            continue
        fn_name = self.grammar.get_name(
            production_key=str(primitive), name_classes=self.function_name_classes
        )
        fn_type = primitive.infer()
        if primitive.isInvented:
            fn_body = str(
                self.grammar.show_program(
                    str(primitive)[
                        1:
                    ],  # Remove leading `#` so that any inlined abstractions are replaced with their fn_name
                    name_classes=[
                        LAPSGrammar.HUMAN_READABLE,
                        LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                    ],
                )
            )
        else:
            fn_body = str(primitive)
        fn_description = self.grammar.get_function_description(str(primitive))
        dsl_fns.append((primitive, fn_name, fn_type, fn_body, fn_description))

    dsl_description = (
        "You are an expert programmer working in a language based on lambda calculus.\n"
        + "Your goal is to write programs that accomplish the tasks specified by the user.\n"
    )
    if "dsl_description_prefix" in self.experiment_state.metadata:
        dsl_description += (
            self.experiment_state.metadata["dsl_description_prefix"] + "\n"
        )

    dsl_description += "\nWrite programs using the available functions:\n\n"

    for primitive, fn_name, fn_type, fn_body, fn_description in dsl_fns:
        docstring = f"{fn_name} :: {fn_type}"
        if primitive.isInvented:
            docstring += f"\n{fn_body}"
        if fn_description is not None:
            docstring += f"\ndescription: {fn_description}"
        dsl_description += docstring + "\n\n"

    return dsl_description


class ClusterSolver(GPTSampleGenerator):
    name = "cluster_solver"

    results_file = "cluster_solver_results.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return ClusterSolver(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(self, engine=engine)

    def construct_initial_prompt(
            self,
        ):
            