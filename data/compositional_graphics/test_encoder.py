"""compositional_graphics: test_encoder | Author : Catherine Wong"""

from src.models.model_loaders import ModelLoaderRegistries, EXAMPLES_ENCODER

from src.experiment_iterator import *
from src.test_experiment_iterator import (
    example_encoder_config_block,
    TEST_GRAPHICS_CONFIG,
)

ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]

import data.compositional_graphics.encoder as to_test

from dreamcoder.domains.logo.main import LogoFeatureCNN


def test_logo_feature_cnn_load_model_initializer():
    model_loader = ExamplesEncoderRegistry[to_test.LogoFeatureCNNExamplesEncoder.name]

    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    model_initializer = model_loader.load_model_initializer(
        **example_encoder_config_block[PARAMS]
    )
    # Try to initialize.
    model = model_initializer(test_experiment_state)

    assert type(model) == LogoFeatureCNN
