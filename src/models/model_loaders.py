"""
model_loaders.py | Author : Catherine Wong

Utility functions for loading and initializing many basic models.
"""
from class_registry import ClassRegistry

GRAMMAR = "grammar"
LANGUAGE_ENCODER = "language_encoder"
EXAMPLES_ENCODER = "examples_encoder"
JOINT_LANGUAGE_EXAMPLES_ENCODER = "joint_language_examples_encoder"
AMORTIZED_SYNTHESIS = "amortized_synthesis"
ModelLoaderRegistries = {
    GRAMMAR: ClassRegistry("name", unique=True),
    EXAMPLES_ENCODER: ClassRegistry("name", unique=True),
    LANGUAGE_ENCODER: ClassRegistry("name", unique=True),
    JOINT_LANGUAGE_EXAMPLES_ENCODER: ClassRegistry("name", unique=True),
    AMORTIZED_SYNTHESIS: ClassRegistry("name", unique=True),
}


class ModelLoader:
    """Abstract class for loading generic models."""

    def load_model(self, experiment_state, **kwargs):
        raise NotImplementedError

    def load_model_from_checkpoint(self, experiment_state, checkpoint_directory):
        raise NotImplementedError
