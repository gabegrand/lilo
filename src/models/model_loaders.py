"""
model_loaders.py | Author : Catherine Wong

Utility functions for loading and initializing many basic models.
"""


class GrammarLoader:
    """Abstract class for loading Grammars."""

    def load_model(self):
        raise NotImplementedError

    def load_model_from_checkpoint(self, checkpoint_directory):
        raise NotImplementedError
