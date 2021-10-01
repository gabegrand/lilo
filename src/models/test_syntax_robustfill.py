"""
test_syntax_robustfill.py | Author : Catherine Wong
"""
import src.models.syntax_robustfill as syntax_robustfill
from src.experiment_iterator import *
from src.models.model_loaders import *
from src.task_loaders import *
from src.test_experiment_iterator import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState

single_image_example_encoder_config_block = {
    MODEL_TYPE: EXAMPLES_ENCODER,
    MODEL_LOADER: syntax_robustfill.SingleImageExampleEncoder.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {"cuda": False},
}
language_encoder_config_block = {
    MODEL_TYPE: LANGUAGE_ENCODER,
    MODEL_LOADER: syntax_robustfill.SequenceLanguageEncoder.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {},
}
syntax_robustfill_config_block = {
    MODEL_TYPE: AMORTIZED_SYNTHESIS,
    MODEL_LOADER: syntax_robustfill.SyntaxRobustfill.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {"task_encoder_types": ["language_encoder"]},
}


TEST_SEQUENCE_CONFIG = TEST_GRAPHICS_CONFIG

# Disable logging.
(
    TEST_SEQUENCE_CONFIG[METADATA][LOG_DIRECTORY],
    TEST_SEQUENCE_CONFIG[METADATA][EXPORT_DIRECTORY],
) = (None, None)

TEST_SEQUENCE_CONFIG[MODEL_INITIALIZERS] = [
    grammar_config_block,
    single_image_example_encoder_config_block,
    language_encoder_config_block,
    syntax_robustfill_config_block,
]
# Image Example encoder tests.
def _get_default_image_encoder():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)
    test_model = syntax_robustfill.SingleImageExampleEncoder(
        experiment_state=test_experiment_state
    )
    return test_experiment_state, test_model


def test_single_image_encoder_forward():
    test_experiment_state, test_model = _get_default_image_encoder()

    test_task_ids = ["a small triangle", "a medium triangle"]
    test_tasks = test_experiment_state.get_tasks_for_ids(
        task_split=TRAIN, task_ids=test_task_ids
    )
    test_images = [t.highresolution for t in test_tasks]
    SingleImageExampleEncoder = syntax_robustfill.SingleImageExampleEncoder
    test_model = SingleImageExampleEncoder(experiment_state=test_experiment_state)
    test_model(test_images)


# Sequence Language Encoder tests.
def _get_default_sequence_encoder(bidirectional=True):
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    SequenceLanguageEncoder = syntax_robustfill.SequenceLanguageEncoder
    test_model = SequenceLanguageEncoder(
        experiment_state=test_experiment_state,
        bidirectional=bidirectional,
        encoder_dim=SequenceLanguageEncoder.DEFAULT_ENCODER_DIM,
    )
    return test_experiment_state, test_model


def test_sequence_language_encoder_token_to_idx():
    SequenceLanguageEncoder = syntax_robustfill.SequenceLanguageEncoder
    test_experiment_state, test_model = _get_default_sequence_encoder()
    train_vocab_len = len(test_experiment_state.task_vocab[TRAIN])

    PAD, UNK, START, END = (
        SequenceLanguageEncoder.PAD,
        SequenceLanguageEncoder.UNK,
        SequenceLanguageEncoder.START,
        SequenceLanguageEncoder.END,
    )
    assert len(test_model.token_to_idx) == train_vocab_len + len([PAD, UNK, START, END])

    assert test_model.token_to_idx[UNK] == 1

    test_unknown_token = "test_unknown_token"
    assert test_model.token_to_idx[test_unknown_token] == 1


def test_sequence_language_encoder_tokenize_padded():
    SequenceLanguageEncoder = syntax_robustfill.SequenceLanguageEncoder
    test_experiment_state, test_model = _get_default_sequence_encoder()
    test_strings, test_lengths = [
        "snowflake",
        "snowflake snowflake snowflake",
        "snowflake snowflake",
    ], [
        base_length + 2 for base_length in [1, 3, 2]
    ]  # Account for START and END

    tokenized_arrays = test_model._batch_tokenize_strings(test_strings)
    for batch_idx, tokenized_array in enumerate(tokenized_arrays):
        assert tokenized_array[0] == SequenceLanguageEncoder.START
        assert tokenized_array[-1] == SequenceLanguageEncoder.END
        assert len(tokenized_array) == test_lengths[batch_idx]

    token_tensor, lengths = test_model._input_strings_to_padded_token_tensor(
        test_strings
    )
    assert list(token_tensor.size()) == [len(test_strings), max(test_lengths)]
    assert list(lengths) == test_lengths


def test_sequence_language_encoder_forward():
    SequenceLanguageEncoder = syntax_robustfill.SequenceLanguageEncoder

    test_strings, test_lengths = [
        "snowflake",
        "snowflake snowflake snowflake",
        "snowflake snowflake",
    ], [
        base_length + 2 for base_length in [1, 3, 2]
    ]  # Account for START and END

    batch_size = len(test_strings)
    seq_length = max(test_lengths)
    dim = SequenceLanguageEncoder.DEFAULT_ENCODER_DIM

    # Unidirectional GRU
    test_experiment_state, test_model = _get_default_sequence_encoder(
        bidirectional=False
    )
    outputs, hidden = test_model(test_strings)
    assert list(outputs.size()) == [
        batch_size,
        seq_length,
        dim,
    ]
    assert list(hidden.size()) == [
        batch_size,
        1,
        dim,
    ]

    # Bidirectional GRU
    test_experiment_state, test_model = _get_default_sequence_encoder(
        bidirectional=True
    )
    outputs, hidden = test_model(test_strings)
    assert list(outputs.size()) == [
        batch_size,
        seq_length,
        dim,
    ]
    assert list(hidden.size()) == [
        batch_size,
        1,
        dim,
    ]


# SyntaxRobustfill model tests.
def test_syntax_robustfill_load_model():
    test_config = TEST_SEQUENCE_CONFIG
    test_experiment_state = ExperimentState(test_config)

    assert (
        type(test_experiment_state.models[AMORTIZED_SYNTHESIS])
        == syntax_robustfill.SyntaxRobustfill
    )


def test_syntax_robustfill_initialize_encoders():
    SyntaxRobustfill = syntax_robustfill.SyntaxRobustfill

    test_config = TEST_SEQUENCE_CONFIG
    test_experiment_state = ExperimentState(test_config)

    task_encoders_and_types = [
        ([LANGUAGE_ENCODER], syntax_robustfill.SequenceLanguageEncoder),
        ([EXAMPLES_ENCODER], syntax_robustfill.SingleImageExampleEncoder),
    ]
    for task_encoder_types, task_encoder_class in task_encoders_and_types:
        test_model = SyntaxRobustfill(
            experiment_state=test_experiment_state,
            task_encoder_types=task_encoder_types,
        )
        assert type(test_model.encoder) == task_encoder_class
