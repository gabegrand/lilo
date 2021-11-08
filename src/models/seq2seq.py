"""
seq2seq.py | Authors : Gabriel Grand & Catherine Wong.

Implements modular sequence-to-sequence NN components for transducing from
language and/or images to programs.

End-to-end wrapper:
- Seq2Seq

Encoders:
- SequenceLanguageEncoder
- ImageExampleEncoder
- JointLanguageImageEncoder (WIP)

Decoders:
- SequenceProgramDecoder
"""
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

import src.models.model_loaders as model_loaders
from src.experiment_iterator import ExperimentState
from src.task_loaders import *

ExamplesEncoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.EXAMPLES_ENCODER
]
LanguageEncoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LANGUAGE_ENCODER
]

ProgramDecoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.PROGRAM_DECODER
]

AmortizedSynthesisModelRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.AMORTIZED_SYNTHESIS
]


@ExamplesEncoderRegistry.register
class SingleImageExampleEncoder(nn.Module, model_loaders.ModelLoader):
    """Image encoder for tasks specified by a single image example.  Reference implementation: LOGO Feature CNN from DreamCoder."""

    name = "single_image_example_encoder"  # String key for config and encoder registry.

    ENCODER_INPUT_DIM = 128  # Expects 128 x 128 image pixels
    ENCODER_HIDDEN_DIM = 64
    ENCODER_INPUT_CHANNELS = 1
    ENCODER_KERNEL_SIZE = 3
    PIXEL_NORMALIZATION_CONSTANT = 256.0

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return SingleImageExampleEncoder(experiment_state=experiment_state, **kwargs)

    def __init__(
        self,
        experiment_state=None,
        encoder_input_dim=ENCODER_INPUT_DIM,
        encoder_input_channels=ENCODER_INPUT_CHANNELS,
        encoder_kernel_size=ENCODER_KERNEL_SIZE,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        endpool=True,
        cuda=False,  # TODO: implement CUDA support.
    ):
        super().__init__()
        self.encoder_input_dim = encoder_input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_input_channels = encoder_input_channels
        self.encoder_kernel_size = encoder_kernel_size
        # Multi-layer CNN: reference: https://github.com/mila-iqia/babyai/blob/master/babyai/model.py

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, self.encoder_kernel_size, padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.image_conv = nn.Sequential(
            conv_block(self.encoder_input_channels, self.encoder_hidden_dim),
            conv_block(self.encoder_hidden_dim, self.encoder_hidden_dim),
            conv_block(self.encoder_hidden_dim, self.encoder_hidden_dim),
            conv_block(self.encoder_hidden_dim, self.encoder_hidden_dim),
            conv_block(self.encoder_hidden_dim, self.encoder_hidden_dim),
            conv_block(self.encoder_hidden_dim, self.encoder_hidden_dim),
            Flatten(),
        )

    def _batch_reshape_and_add_channels_to_tensor(
        self, images, height=128, width=128, n_channels=1
    ):
        """
        inputs: [nbatch flattened single channel images]
        output: [nbatch * C_in, H, W] images.
        """
        images = torch.tensor(images, dtype=torch.float)
        images = torch.reshape(images, (-1, n_channels, height, width))
        # # images = torch.transpose(torch.transpose(images, 1, 3), 2, 3)
        return images

    def forward(self, inputs):
        """inputs: [n_batch images].
        output: [n_batch * (ENCODER_INPUT_DIM * 2)] encoding.
        """
        if len(inputs[0]) == self.encoder_input_dim * self.encoder_input_dim:
            image_tensor = self._batch_reshape_and_add_channels_to_tensor(inputs)
        else:
            image_tensor = torch.tensor(inputs)
        # Normalize by pixels
        image_tensor /= self.PIXEL_NORMALIZATION_CONSTANT
        examples_embedding = self.image_conv(image_tensor)
        return examples_embedding


@LanguageEncoderRegistry.register
class SequenceLanguageEncoder(nn.Module, model_loaders.ModelLoader):
    """Language encoder for sequences of language tokens."""

    name = "sequence_language_encoder"  # String key for config and encoder registry.

    DEFAULT_EMBEDDING_DIM = 128
    DEFAULT_ENCODER_DIM = 128
    START, END, UNK, PAD = "<START>", "<END>", "<UNK>", "<PAD>"
    WORD_TOKENIZE = "word_tokenize"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return SequenceLanguageEncoder(experiment_state=experiment_state, **kwargs)

    def __init__(
        self,
        experiment_state=None,
        bidirectional=True,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        encoder_dim=DEFAULT_ENCODER_DIM,
        tokenizer_fn=WORD_TOKENIZE,
        cuda=False,  # TODO: implement CUDA support.
    ):
        super().__init__()

        if experiment_state is None:
            """
            If experiment_state is None, we cannot init the model.
            """
            return None

        self.tokenizer_fn, self.tokenizer_cache = self._init_tokenizer(tokenizer_fn)
        self.token_to_idx = self._init_token_to_idx_from_experiment_state(
            experiment_state
        )

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=len(self.token_to_idx),
            embedding_dim=self.embedding_dim,
            padding_idx=self.token_to_idx[self.PAD],
        )

        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        assert encoder_dim % self.num_directions == 0
        self.encoder_dim = encoder_dim // self.num_directions
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.encoder_dim,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def _init_tokenizer(self, tokenizer_fn):
        tokenizer_cache = dict()
        if tokenizer_fn == self.WORD_TOKENIZE:
            from nltk.tokenize import word_tokenize

            return word_tokenize, tokenizer_cache
        else:
            raise NotImplementedError(
                "WORD_TOKENIZE is the only supported tokenizer_fn."
            )

    def _init_token_to_idx_from_experiment_state(self, experiment_state):
        """Initialize the token_to_idx from the experiment state. This default dictionary also returns the UNK token for any unfound tokens"""
        if experiment_state == None:
            return {}
        train_vocab = sorted(list(experiment_state.task_vocab[TRAIN]))
        train_vocab = [self.PAD, self.UNK, self.START, self.END] + train_vocab

        token_to_idx = defaultdict(lambda: 1)  # Default index 1 -> UNK; 0 is padding
        for token_idx, token in enumerate(train_vocab):
            token_to_idx[token] = token_idx

        return token_to_idx

    def _batch_tokenize_strings(self, inputs):
        """inputs: [n_batch input strings].
        outputs: [n_batch [START + tokens + END] token arrays]."""
        tokenized_strings = []
        for input_string in inputs:
            if input_string in self.tokenizer_cache:
                tokenized_strings.append(self.tokenizer_cache[input_string])
            else:
                tokenized_string = (
                    [self.START] + self.tokenizer_fn(input_string) + [self.END]
                )
                self.tokenizer_cache[input_string] = tokenized_string
                tokenized_strings.append(tokenized_string)
        return tokenized_strings

    def _input_strings_to_padded_token_tensor(self, inputs):
        """ TODO(gg): Replace with torch.nn.utils.rnn.pad_sequence"""
        """inputs: [n_batch input strings].
        :ret: [n_batch * max_token_len] tensor padded with 0s; lengths.
        """
        input_token_arrays = self._batch_tokenize_strings(inputs)
        max_len = max([len(s) for s in input_token_arrays])
        input_token_indices, lengths = [], []
        for input_token_array in input_token_arrays:
            token_length = len(input_token_array)
            lengths.append(token_length)
            input_token_index_array = [
                self.token_to_idx[t] for t in input_token_array
            ] + [self.token_to_idx[self.PAD]] * (max_len - token_length)
            input_token_indices.append(input_token_index_array)
        input_token_indices, lengths = torch.tensor(input_token_indices), torch.tensor(
            lengths
        )
        return input_token_indices, lengths

    def forward(self, inputs, hidden=None):
        """Encoder forward pass.

        :params:
            inputs: List [batch_size] of space-separated input strings.
            hidden: Tensor of size [batch_size, 1, self.num_directions * self.encoder_dim].
                If None, `hidden` will be auto-initialized.

        :returns:
            outputs: [batch_size, seq_len, self.num_directions * self.encoder_dim]
            hidden: [batch_size, 1, self.num_directions * self.encoder_dim]

        """
        batch_size = len(inputs)
        if hidden is not None:
            assert hidden.size() == [
                batch_size,
                1,
                self.num_directions * self.encoder_dim,
            ]

        padded_tokens, token_lengths = self._input_strings_to_padded_token_tensor(
            inputs
        )
        embedded = self.embedding(padded_tokens)
        packed = pack_padded_sequence(
            embedded, token_lengths, batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        hidden = hidden.view(-1, 1, self.num_directions * self.encoder_dim)
        return outputs, hidden


@ProgramDecoderRegistry.register
class SequenceProgramDecoder(nn.Module, model_loaders.ModelLoader):
    """TODO(gg): Refactor with SequenceLanguageEncoder to inherit shared superclass.

    Adapted from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

    """

    name = "sequence_program_decoder"

    DEFAULT_DECODER_DIM = 128
    DEFAULT_ATTENTION_DIM = 128
    DEFAULT_ATTN_MODEL = "dot"
    DEFAULT_DROPOUT_P = 0
    MAX_SEQUENCE_LENGTH = 128  # TODO(gg): Verify that this value makes sense
    CLIP_GRAD_MAX_NORM = 50.0  # TODO(gg): Verify that this value makes sense
    START, END, UNK, PAD = "<START>", "<END>", "<UNK>", "<PAD>"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return SequenceProgramDecoder(experiment_state=experiment_state, **kwargs)

    def __init__(
        self,
        experiment_state=None,
        decoder_dim=DEFAULT_DECODER_DIM,
        attn_model=DEFAULT_ATTN_MODEL,
        dropout_p=DEFAULT_DROPOUT_P,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        clip_grad_max_norm=CLIP_GRAD_MAX_NORM,
        cuda=False,  # TODO: implement CUDA support.
    ):
        super().__init__()

        if experiment_state is None:
            """
            If experiment_state is None, we cannot init the model.
            """
            return None

        self.token_to_idx = self._init_token_to_idx_from_experiment_state(
            experiment_state
        )

        self.attn_model = attn_model
        self.hidden_size = decoder_dim
        self.output_size = len(self.token_to_idx)
        self.dropout = dropout_p
        self.max_sequence_length = max_sequence_length
        self.clip_grad_max_norm = clip_grad_max_norm

        # Define layers
        self.embedding = nn.Embedding(
            num_embeddings=self.output_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.token_to_idx[self.PAD],
        )
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=self.dropout,
        )
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Choose attention model
        self.attn = DecoderAttn(self.attn_model, self.hidden_size)

    def _init_token_to_idx_from_experiment_state(self, experiment_state):
        """Initialize the token_to_idx from the experiment state. This default dictionary also returns the UNK token for any unfound tokens"""
        if experiment_state == None:
            return {}
        train_vocab = sorted(list(experiment_state.models[model_loaders.GRAMMAR].vocab))
        train_vocab = [self.PAD, self.UNK, self.START, self.END] + train_vocab

        token_to_idx = defaultdict(lambda: 1)  # Default index 1 -> UNK; 0 is padding
        for token_idx, token in enumerate(train_vocab):
            token_to_idx[token] = token_idx

        return token_to_idx

    def _batch_tokenize_strings(self, inputs):
        """Unlike in the encoder, inputs are already tokenized.

        inputs: [n_batch [tokens] arrays].
        outputs: [n_batch [START + tokens + END] token arrays].

        """
        return [[self.START] + token_list + [self.END] for token_list in inputs]

    def _input_strings_to_padded_token_tensor(self, inputs):
        """ TODO(gg): Replace with torch.nn.utils.rnn.pad_sequence"""
        """inputs: [n_batch [tokens] arrays].
        :ret: [n_batch * max_token_len] tensor padded with 0s; lengths.
        """
        input_token_arrays = self._batch_tokenize_strings(inputs)
        max_len = max([len(s) for s in input_token_arrays])
        input_token_indices, lengths = [], []
        for input_token_array in input_token_arrays:
            token_length = len(input_token_array)
            lengths.append(token_length)
            input_token_index_array = [
                self.token_to_idx[t] for t in input_token_array
            ] + [self.token_to_idx[self.PAD]] * (max_len - token_length)
            input_token_indices.append(input_token_index_array)
        input_token_indices, lengths = torch.tensor(input_token_indices), torch.tensor(
            lengths
        )
        return input_token_indices, lengths

    def forward(self, last_token_idx, last_hidden, encoder_outputs):
        """Decoder forward pass.

        NOTE: Unlike the encoder, the decoder accepts a *single* token at a time
        and is designed to run in the context of a for loop. Also, note that
        `last_token_idx` takes a target token *index* (as produced by
        _input_strings_to_padded_token_tensor), and not a token string.

        :params:
            last_token_idx: Tensor [batch_size] of token idxs corresponding to
                the decoder output from the previous timestep.
            last_hidden: Tensor [batch_size, 1, self.hidden_size] corresponding
                to the decoder hidden state from the previous timestep.
            encoder_outputs: Tensor [batch_size, seq_len, self.hidden_size] of
                encoder outputs over which to perform attention.

        :returns:
            output: Tensor [batch_size, self.output_size]
            hidden: Tensor [batch_size, 1, self.hidden_size]
            att_weights: Tensor [batch_size, seq_len, 1]

        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        assert last_token_idx.size() == (batch_size,), last_token_idx.size()
        assert last_hidden.size() == (
            batch_size,
            1,
            self.hidden_size,
        ), last_hidden.size()
        assert encoder_outputs.size() == (
            batch_size,
            seq_len,
            self.hidden_size,
        ), encoder_outputs.size()  # Sequence length unknown

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(last_token_idx)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(batch_size, 1, self.hidden_size)

        # nn.gru expects hidden.size() = (1, batch, dim) regardless of `batch_first=True`
        last_hidden = last_hidden.view(1, batch_size, self.hidden_size)

        # Get current hidden state from input word and last hidden state
        decoder_output, hidden = self.gru(embedded, last_hidden)

        # Revert back to batch-first after call to gru()
        # hidden.size() = (batch, 1, dim)
        hidden = hidden.view(batch_size, 1, self.hidden_size)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(decoder_output, encoder_outputs)

        # (batch, 1, seq) x (batch, seq, dim) = (batch, 1, dim)
        context = attn_weights.view(batch_size, 1, seq_len).bmm(
            encoder_outputs
        )  # B x 1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        decoder_output = decoder_output.squeeze(1)  # B x 1 x N -> B x N
        context = context.squeeze(1)  # B x 1 x N -> B x N
        concat_input = torch.cat((decoder_output, context), dim=1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        assert output.size() == (batch_size, self.output_size), output.size()
        assert hidden.size() == (batch_size, 1, self.hidden_size), hidden.size()
        assert attn_weights.size() == (batch_size, seq_len, 1)

        return output, hidden, attn_weights


class DecoderAttn(nn.Module):
    """From https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb"""

    ATTN_DOT, ATTN_GENERAL, ATTN_CONCAT = "dot", "general", "concat"

    def __init__(self, method, hidden_size, cuda=False):
        super(DecoderAttn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.cuda = cuda

        if self.method == self.ATTN_DOT:
            pass  # No changes needed
        elif self.method == self.ATTN_GENERAL:
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == self.ATTN_CONCAT:
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            raise ValueError(f"Unknown attention method: {self.method}")

    def forward(self, decoder_output, encoder_outputs):
        this_batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if self.cuda:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(
                    decoder_output[b], encoder_outputs[b, i].unsqueeze(0)
                )

        # Normalize energies to weights in range 0 to 1, resize to B x S x 1
        return F.softmax(attn_energies, dim=1).unsqueeze(-1)

    def score(self, decoder_output, encoder_output):
        if self.method == self.ATTN_DOT:
            energy = decoder_output.squeeze().dot(encoder_output.squeeze())
            return energy

        elif self.method == self.ATTN_GENERAL:
            energy = self.attn(encoder_output)
            energy = decoder_output.squeeze().dot(energy.squeeze())
            return energy

        elif self.method == self.ATTN_CONCAT:
            energy = self.attn(torch.cat((decoder_output, encoder_output), dim=1))
            energy = self.v.dot(energy.squeeze())
            return energy

        else:
            raise ValueError(f"Unknown attention method: {self.method}")


# Seq2Seq model.
@AmortizedSynthesisModelRegistry.register
class Seq2Seq(nn.Module, model_loaders.ModelLoader):
    """Sequence-to-sequence wrapper for encoder/decoder models."""

    name = "seq2seq"

    DECODER_HIDDEN_SIZE = 512

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return Seq2Seq(experiment_state=experiment_state, **kwargs)

    def __init__(
        self,
        experiment_state=None,
        task_encoder_types=[model_loaders.LANGUAGE_ENCODER],
        decoder_hidden_size=DECODER_HIDDEN_SIZE,
        reset_random_parameters=True,  # Reset to random initialization
    ):
        super().__init__()

        self.task_encoder_types = task_encoder_types
        self.encoder = None
        self.decoder = None
        self.decoder_hidden_size = decoder_hidden_size

        self._initialize_encoders(experiment_state, task_encoder_types)
        self._initialize_decoder(experiment_state)

    # Helper attribute getters.
    def _use_language(self):
        return model_loaders.LANGUAGE_ENCODER in task_encoder_types

    def _use_examples(self):
        return model_loaders.EXAMPLES_ENCODER in task_encoder_types

    # Initialization for the model architecture.
    def _initialize_encoders(self, experiment_state, task_encoder_types):
        """Reinitializes the encoders in the experiment_state. Mutates experiment_state.models to all of the task encoders in task_encoder_types."""
        if experiment_state is None:
            return

        experiment_state.init_models_from_config(
            config=experiment_state.config,
            models_to_initialize=[
                t
                for t in task_encoder_types
                if t != model_loaders.JOINT_LANGUAGE_EXAMPLES_ENCODER
            ],
        )

        # Set the encoder if we haven't already.
        if model_loaders.LANGUAGE_ENCODER in task_encoder_types:
            self.encoder = experiment_state.models[model_loaders.LANGUAGE_ENCODER]

        if model_loaders.EXAMPLES_ENCODER in task_encoder_types:
            self.encoder = experiment_state.models[model_loaders.EXAMPLES_ENCODER]

        # Initialize the joint encoder to determine how to combine the embeddings.
        if model_loaders.JOINT_LANGUAGE_EXAMPLES_ENCODER in task_encoder_types:
            # TODO: implement
            raise NotImplementedError()

    def _initialize_decoder(self, experiment_state):
        # Initialize decoder.

        if experiment_state is None:
            return

        experiment_state.init_models_from_config(
            config=experiment_state.config,
            models_to_initialize=[model_loaders.PROGRAM_DECODER],
        )

        self.decoder = experiment_state.models[model_loaders.PROGRAM_DECODER]

    def _encode_tasks(self, task_split, task_ids, experiment_state):
        # Forward pass encoding of the inputs. This should encode the inputs according to the language, images, or both using the self.encoder

        # TODO(gg): implement this to encode the tasks according to the task language, which is extracted below.
        if self._use_language:
            # Nested list: [[task_0_tokens_0, task_0_tokens_1, ...], [task_1_tokens_0, task_1_tokens_1, ...], ...]
            language_for_ids = experiment_state.get_language_for_ids(
                task_split, task_ids
            )
            n_inputs_per_task = torch.LongTensor([len(x) for x in language_for_ids])

            # Flattened list: [task_0_tokens_0, task_0_tokens_1, ..., task_1_tokens_0, task_1_tokens_1, ...]
            language_flattened = [
                token_string
                for task_language_list in language_for_ids
                for token_string in task_language_list
            ]

            encoder_outputs, encoder_hidden = self.encoder(language_flattened)

            return encoder_outputs, encoder_hidden, n_inputs_per_task

        # TODO(gg): Implement for images and joint cases
        raise NotImplementedError()

    def _decode_tasks(self, last_token_idx, encoder_outputs, encoder_hidden):
        return self.decoder(
            last_token_idx=last_token_idx,
            last_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
        )

    def _run_tasks(
        self,
        task_split: str,
        task_batch_ids: List[str],
        experiment_state: ExperimentState,
        encoder_optimizer: torch.optim.Optimizer = None,
        decoder_optimizer: torch.optim.Optimizer = None,
        mode: str = TRAIN,
    ):
        """Runs the full encoder/decoder model on a set of tasks and computes
        the cross entropy loss.

        Since each task may contain multiple language descriptions and also
        multipe program annotations, we run the model on a set of examples
        generated by taking the per-task cross product of inputs and outputs:

            task_examples = task_inputs x task_outputs

        :params:
            task_split: TRAIN or TEST. Used only to `get_frontiers_for_ids`.
            task_batch_ids: List of task_name to run on.
            experiment_state: ExperimentState instance.
            encoder_optimizer: torch.optim.Optimizer instance.
            decoder_optimizer: torch.optim.Optimizer instance.
            mode: TRAIN or TEST. If TRAIN, updates model weights; otherwise,
                only runs inference and computes loss.

        :returns:
            {
                "loss": Scalar loss value, averaged across all tasks. Note
                    that tasks are not necessarily equally weighted, since
                    some tasks may have more associated language descriptions
                    or programs and therefore generate more examples.
                "loss_per_task": Dict { task_name : task_loss } where loss
                    is averaged over all examples associated with a task.
                "n_tasks": Number of tasks retrieved by `get_frontiers_for_ids`.
                "n_inputs_per_task": List of length n_tasks with number of
                    language descriptions per task.
                "n_outputs_per_task": List of length n_tasks with number of
                    program annotations per task.
                "n_examples_per_task": List of length n_tasks with number of
                    generated examples per task. This is equal to the
                    pointwise product of `n_inputs_per_task * n_outputs_per_task`.
            }

            NOTE: If none of the frontiers have any entries (no solved programs),
            then it is not possible to compute a loss. In this case, `_run_tasks`
            will return `None` and is up to the caller to handle this corner
            case appropriately.
        """
        if mode == TRAIN:
            self.encoder.train()
            self.decoder.train()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
        elif mode == TEST:
            self.encoder.eval()
            self.decoder.eval()
        else:
            raise ValueError(mode)

        frontiers = experiment_state.get_frontiers_for_ids(
            task_split=task_split, task_ids=task_batch_ids
        )

        # Remove frontiers with no programs
        frontiers = [f for f in frontiers if len(f.entries) > 0]
        if len(frontiers) == 0:
            return None

        # Ground truth program tokens for supervision
        target_tokens = [e.tokens for f in frontiers for e in f.entries]
        n_outputs_per_task = torch.LongTensor([len(f.entries) for f in frontiers])
        (
            target_token_idxs,
            target_lengths,
        ) = self.decoder._input_strings_to_padded_token_tensor(target_tokens)

        loss = 0

        encoder_outputs, encoder_hidden, n_inputs_per_task = self._encode_tasks(
            task_split, task_batch_ids, experiment_state
        )

        # Construct cross-product for encoder_outputs
        repeats_encoder_outputs = torch.repeat_interleave(
            input=n_outputs_per_task, repeats=n_inputs_per_task
        )
        encoder_outputs = torch.repeat_interleave(
            input=encoder_outputs, repeats=repeats_encoder_outputs, dim=0
        )

        # Construct cross-product for target_token_idxs
        repeats_target_token_idxs = torch.repeat_interleave(
            input=n_inputs_per_task, repeats=n_outputs_per_task
        )
        target_token_idxs = torch.repeat_interleave(
            input=target_token_idxs, repeats=repeats_target_token_idxs, dim=0
        )

        assert encoder_outputs.size(0) == target_token_idxs.size(0)
        batch_size = encoder_outputs.size(0)
        target_seq_len = target_token_idxs.size(1)
        n_examples_per_task = n_inputs_per_task * n_outputs_per_task

        # Train using teacher forcing
        # TODO(gg): Implement scheduled sampling
        all_decoder_outputs = Variable(
            torch.zeros(batch_size, target_seq_len, self.decoder.output_size)
        )

        for t in range(target_seq_len):
            decoder_input = target_token_idxs[:, t]  # Next input is current target
            decoder_output, decoder_hidden, decoder_attn = self._decode_tasks(
                decoder_input, encoder_outputs, encoder_hidden
            )
            all_decoder_outputs[:, t, :] = decoder_output

        # Loss calculation and backpropagation
        loss_per_example = nn.functional.cross_entropy(
            input=all_decoder_outputs.view(
                batch_size, self.decoder.output_size, target_seq_len
            ),
            target=target_token_idxs,
            reduction="none",
            ignore_index=self.decoder.token_to_idx[self.decoder.PAD],
        )
        loss = loss_per_example.mean()

        # Compute mean loss per task = inputs x outputs within loss_per_example
        loss_per_task = {}
        for i, f in enumerate(frontiers):
            example_idx_start, example_idx_stop = sum(n_examples_per_task[:i]), sum(
                n_examples_per_task[: i + 1]
            )
            task_loss = (
                loss_per_example[example_idx_start:example_idx_stop].mean().item()
            )
            loss_per_task[f.task.name] = task_loss

        if mode == TRAIN:
            loss.backward()

            # Clip gradient norms
            ec = nn.utils.clip_grad_norm_(
                parameters=self.encoder.parameters(),
                max_norm=self.decoder.clip_grad_max_norm,
            )
            dc = nn.utils.clip_grad_norm_(
                parameters=self.decoder.parameters(),
                max_norm=self.decoder.clip_grad_max_norm,
            )

            # Update parameters with optimizers
            encoder_optimizer.step()
            decoder_optimizer.step()

        return {
            "loss": loss.item(),
            "loss_per_task": loss_per_task,
            "n_tasks": len(frontiers),
            "n_inputs_per_task": n_inputs_per_task.tolist(),
            "n_outputs_per_task": n_outputs_per_task.tolist(),
            "n_examples_per_task": n_examples_per_task.tolist(),
        }

    def optimize_model_for_frontiers(
        self,
        experiment_state,
        task_split=TRAIN,
        task_batch_ids=ALL,
        recognition_train_epochs=100,  # Max epochs to fit the model
        learning_rate=1e-2,
        early_stopping_epsilon=1e-4,
        early_stopping_patience=5,
    ):
        """Train the model with respect to the tasks in task_batch_ids.
        The model is trained to regress from a task encoding according to
        task_encoder_types (either text, images, or a joint encoding of both) to
        predict corresponding programs for each task.

        :params:
            experiment_state: ExperimentState object.
            task_split: which split to train tasks on.
            task_batch_ids: list of IDs of tasks to train on or ALL to train on
                all possible solved tasks in the split.
            early_stopping_epsilon: Best train loss must decrease by at least this
                amount each epoch, otherwise the early stopping counter is
                incremented.
            early_stopping_patience: If the train loss doesn't decrease by
                early_stopping_epsilon for this number of consecutive epochs,
                then early stopping will be triggered. Note that the counter
                resets after each "successful" training epoch in which the
                conditions are not met.

        On completion, model parameters should be updated to the trained model.
        """
        encoder_optimizer = Adam(params=self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = Adam(params=self.decoder.parameters(), lr=learning_rate)

        run_results_per_epoch = []
        loss_per_epoch = []
        early_stopping_counter = (
            0  # Tracks the number of times early stopping conditions were met.
        )

        # TODO(gg): Implement batching over task ids
        for epoch in range(recognition_train_epochs):
            run_results = self._run_tasks(
                task_split,
                task_batch_ids,
                experiment_state,
                encoder_optimizer,
                decoder_optimizer,
                mode=TRAIN,
            )

            if run_results is None:
                print(
                    f"[TRAIN] Skipped training - None of the frontiers had any entries to train on."
                )
                return None

            run_results["epoch"] = epoch
            run_results_per_epoch.append(run_results)

            loss = run_results["loss"]
            print(
                f"[TRAIN {epoch} / {recognition_train_epochs}] Fit {self.name} on {run_results['n_tasks']} tasks with total loss: {loss}"
            )

            # Check whether to trigger early stopping
            loss_per_epoch.append(loss)

            if epoch > 0:
                best_loss = min(loss_per_epoch[:-1])
                if (best_loss - loss) < early_stopping_epsilon:
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0

                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        return run_results_per_epoch

    def score_frontier_avg_conditional_log_likelihoods(
        self, experiment_state, task_split=TRAIN, task_batch_ids=ALL
    ):
        """
        Evaluates score(frontier for task_id) = mean [log_likelihood(program)
        for program in frontier] where

        log_likelihood = log p(program | task, model_parameters) where program
        is a proposed program solving the task, task is the encoding of the task
        under the model (eg. language, images, or a joint encoding) and
        model_parameters are wrt. a trained model.

        :ret: {
            task_id : score(frontier for task_id)
        }
        """
        # TODO(gg): Implement batching over task ids
        run_results = self._run_tasks(
            task_split,
            task_batch_ids,
            experiment_state,
            mode=TEST,
        )
        if run_results is None:
            raise ValueError(
                f"[EVAL] None of the frontiers had any entries to eval on."
            )

        print(
            f"[TEST] Evaluated {self.name} on {run_results['n_tasks']} tasks with total loss: {run_results['loss']}"
        )
        return run_results


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
