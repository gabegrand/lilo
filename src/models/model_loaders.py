"""
model_loaders.py | Author : Catherine Wong

Utility functions for loading and initializing many basic models.
"""
from class_registry import ClassRegistry

GRAMMAR = "grammar"
INITIALIZE_GROUND_TRUTH = "initialize_ground_truth_task_frontiers"
LIBRARY_NAMER = "library_namer"  # Models that assign names to grammar functions.
LIBRARY_LEARNER = "library_learner"  # Models that update grammars.
PROGRAM_REWRITER = "program_rewriter"  # Models that rewrite programs wrt. grammars
SAMPLE_GENERATOR = (
    "sample_generator"  # Models that generate sample programs, language, or both.
)
LLM_SOLVER = "llm_solver"
LANGUAGE_ENCODER = "language_encoder"
EXAMPLES_ENCODER = "examples_encoder"
JOINT_LANGUAGE_EXAMPLES_ENCODER = "joint_language_examples_encoder"
PROGRAM_DECODER = "program_decoder"
AMORTIZED_SYNTHESIS = "amortized_synthesis"
ModelLoaderRegistries = {
    GRAMMAR: ClassRegistry("name", unique=True),
    LIBRARY_LEARNER: ClassRegistry("name", unique=True),
    LIBRARY_NAMER: ClassRegistry("name", unique=True),
    PROGRAM_REWRITER: ClassRegistry("name", unique=True),
    SAMPLE_GENERATOR: ClassRegistry("name", unique=True),
    LLM_SOLVER: ClassRegistry("name", unique=True),
    EXAMPLES_ENCODER: ClassRegistry("name", unique=True),
    LANGUAGE_ENCODER: ClassRegistry("name", unique=True),
    JOINT_LANGUAGE_EXAMPLES_ENCODER: ClassRegistry("name", unique=True),
    PROGRAM_DECODER: ClassRegistry("name", unique=True),
    AMORTIZED_SYNTHESIS: ClassRegistry("name", unique=True),
}


class ModelLoader:
    """Abstract class for loading generic models."""

    def load_model(self, experiment_state, **kwargs):
        raise NotImplementedError

    def load_model_from_checkpoint(self, experiment_state, checkpoint_directory):
        raise NotImplementedError
