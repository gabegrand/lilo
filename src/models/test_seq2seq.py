"""
test_seq2seq.py | Author : Catherine Wong
"""
import pytest

import src.models.seq2seq as seq2seq
from src.experiment_iterator import *
from src.models.model_loaders import *
from src.task_loaders import *
from src.test_experiment_iterator import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState

single_image_example_encoder_config_block = {
    MODEL_TYPE: EXAMPLES_ENCODER,
    MODEL_LOADER: seq2seq.SingleImageExampleEncoder.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {"cuda": False},
}
language_encoder_config_block = {
    MODEL_TYPE: LANGUAGE_ENCODER,
    MODEL_LOADER: seq2seq.SequenceLanguageEncoder.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {},
}
program_decoder_config_block = {
    MODEL_TYPE: PROGRAM_DECODER,
    MODEL_LOADER: seq2seq.SequenceProgramDecoder.name,
    MODEL_INITIALIZER_FN: "load_model",
    PARAMS: {},
}
seq2seq_config_block = {
    MODEL_TYPE: AMORTIZED_SYNTHESIS,
    MODEL_LOADER: seq2seq.Seq2Seq.name,
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
    seq2seq_config_block,
    program_decoder_config_block,
]
# Image Example encoder tests.
def _get_default_image_encoder():
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)
    test_model = seq2seq.SingleImageExampleEncoder(
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
    SingleImageExampleEncoder = seq2seq.SingleImageExampleEncoder
    test_model = SingleImageExampleEncoder(experiment_state=test_experiment_state)
    test_model(test_images)


# Sequence Language Encoder tests.
def _get_default_sequence_encoder(bidirectional=True):
    test_config = TEST_GRAPHICS_CONFIG
    test_experiment_state = ExperimentState(test_config)

    SequenceLanguageEncoder = seq2seq.SequenceLanguageEncoder
    test_model = SequenceLanguageEncoder(
        experiment_state=test_experiment_state,
        bidirectional=bidirectional,
        encoder_dim=SequenceLanguageEncoder.DEFAULT_ENCODER_DIM,
    )
    return test_experiment_state, test_model


def test_sequence_language_encoder_token_to_idx():
    SequenceLanguageEncoder = seq2seq.SequenceLanguageEncoder
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
    SequenceLanguageEncoder = seq2seq.SequenceLanguageEncoder
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
    SequenceLanguageEncoder = seq2seq.SequenceLanguageEncoder

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


# Seq2Seq model tests.
def test_seq2seq_load_model():
    test_config = TEST_SEQUENCE_CONFIG
    test_experiment_state = ExperimentState(test_config)

    assert type(test_experiment_state.models[AMORTIZED_SYNTHESIS]) == seq2seq.Seq2Seq


def test_seq2seq_optimize_and_score():
    test_config = TEST_SEQUENCE_CONFIG
    test_experiment_state = ExperimentState(test_config)
    test_task_ids = ["a small triangle", "a medium triangle"]
    model = test_experiment_state.models[AMORTIZED_SYNTHESIS]

    # No ground truth, so we expect optimize to return `None`
    train_results = model.optimize_model_for_frontiers(
        test_experiment_state, task_split=TRAIN, task_batch_ids=test_task_ids
    )
    assert train_results is None
    # No ground truth, so we expect score to raise an error
    # (it doesn't make sense to ask for a score when we have no entries)
    with pytest.raises(ValueError):
        eval_results = model.score_frontier_avg_conditional_log_likelihoods(
            test_experiment_state, task_split=TRAIN, task_batch_ids=test_task_ids
        )

    # Now initialize with ground truth - both methods should return a value
    test_experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    train_results = model.optimize_model_for_frontiers(
        test_experiment_state, task_split=TRAIN, task_batch_ids=test_task_ids
    )
    assert train_results
    eval_results = model.score_frontier_avg_conditional_log_likelihoods(
        test_experiment_state, task_split=TRAIN, task_batch_ids=test_task_ids
    )
    assert eval_results


def test_seq2seq_initialize_encoders():
    Seq2Seq = seq2seq.Seq2Seq

    test_config = TEST_SEQUENCE_CONFIG
    test_experiment_state = ExperimentState(test_config)

    task_encoders_and_types = [
        ([LANGUAGE_ENCODER], seq2seq.SequenceLanguageEncoder),
        ([EXAMPLES_ENCODER], seq2seq.SingleImageExampleEncoder),
    ]
    for task_encoder_types, task_encoder_class in task_encoders_and_types:
        test_model = Seq2Seq(
            experiment_state=test_experiment_state,
            task_encoder_types=task_encoder_types,
        )
        assert type(test_model.encoder) == task_encoder_class
