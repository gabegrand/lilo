"""
syntax_robustfill.py | Author : Catherine Wong.

Implements a syntax-aware decoder over program sequences. 
This implementation draws heavily from the syntax_robustfill model implemented in: https://github.com/insperatum/pinn/blob/master/syntax_robustfill.py. 
and the sequence to sequence model implementation in:
https://github.com/mila-iqia/babyai/tree/master/babyai 


This also implements several common front-end encoders designed for user with the sequence decoder: a SequenceLanguageEncoder, ImageExampleEncoder, and JointLanguageImageEncoder.
"""
import nltk
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.task_loaders import *
import src.models.model_loaders as model_loaders


ExamplesEncoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.EXAMPLES_ENCODER
]

# Input encoders.
class SequenceLanguageEncoder(nn.Module):
    """Language encoder for sequences of language tokens. Supports GRU, BIGRU, and ATT_GRU. Reference implementation: https://github.com/mila-iqia/babyai/blob/master/babyai/model.py"""

    GRU, BIGRU, ATT_GRU = "gru", "bigru", "att_gru"
    DEFAULT_ENCODER_DIM = 128
    DEFAULT_ATTENTION_DIM = 128
    START, END, UNK, PAD = "<START>", "<END>", "<UNK>", "<PAD>"
    WORD_TOKENIZE = "word_tokenize"

    def __init__(
        self,
        experiment_state,
        encoder_type=ATT_GRU,
        encoder_dim=DEFAULT_ENCODER_DIM,
        attention_dim=DEFAULT_ATTENTION_DIM,
        tokenizer_fn=WORD_TOKENIZE,
    ):
        super().__init__()

        self.tokenizer_fn, self.tokenizer_cache = self._init_tokenizer(tokenizer_fn)

        self.input_token_to_idx = self._init_input_token_to_idx_from_experiment_state(
            experiment_state
        )
        self.encoder_dim = encoder_dim
        self.input_embedding = nn.Embedding(
            len(self.input_token_to_idx), self.encoder_dim
        )
        self.attention_dim = 2 * attention_dim  # Bidirectional

        self.encoder_type = encoder_type
        if self.encoder_type in [self.GRU, self.BIGRU, self.ATT_GRU]:
            gru_dim = self.encoder_dim
        if self.encoder_type in [self.BIGRU, self.ATT_GRU]:
            gru_dim //= 2
        self.encoder_rnn = nn.GRU(
            self.encoder_dim,
            gru_dim,
            batch_first=True,
            bidirectional=(self.encoder_type in [self.BIGRU, self.ATT_GRU]),
        )
        self.final_encoder_dim = self.encoder_dim

        if self.encoder_type == self.ATT_GRU:
            self.att2key = nn.Linear(self.attention_dim, self.final_encoder_dim)

    def _init_tokenizer(self, tokenizer_fn):
        tokenizer_cache = dict()
        if tokenizer_fn == self.WORD_TOKENIZE:
            from nltk.tokenize import word_tokenize

            return word_tokenize, tokenizer_cache
        else:
            assert False

    def _init_input_token_to_idx_from_experiment_state(self, experiment_state):
        """Initialize the token_to_idx from the experiment state. This default dictionary also returns the UNK token for any unfound tokens"""
        train_vocab = sorted(list(experiment_state.task_vocab[TRAIN]))
        train_vocab = [self.PAD, self.UNK, self.START, self.END] + train_vocab

        input_token_to_idx = defaultdict(
            lambda: 1
        )  # Default index 1 -> UNK; 0 is padding
        for token_idx, token in enumerate(train_vocab):
            input_token_to_idx[token] = token_idx

        return input_token_to_idx

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
                self.input_token_to_idx[t] for t in input_token_array
            ] + [self.input_token_to_idx[self.PAD]] * (max_len - token_length)
            input_token_indices.append(input_token_index_array)
        input_token_indices, lengths = torch.tensor(input_token_indices), torch.tensor(
            lengths
        )
        return input_token_indices, lengths

    def _padded_token_tensor_to_rnn_embeddings(self, padded_tokens, lengths):
        """padded_tokens, lengths = [n_batch * max_len] tensor of token indices; lengths = [n_batch array of lengths of unpadded sequences]
        :ret:
            ATTGRU: n_batch x L x embedding_dim
            GRU, BIGRU: n_batch x embedding_dim tensor of embeddings run through the forward GRU
        """
        if self.encoder_type == self.GRU:
            out, _ = self.encoder_rnn(self.input_embedding(padded_tokens))
            hidden = out[range(len(lengths)), lengths - 1, :]
            return hidden

        elif self.encoder_type in [self.BIGRU, self.ATT_GRU]:
            masks = (padded_tokens != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if padded_tokens.is_cuda:
                    iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.input_embedding(padded_tokens)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(
                    inputs, seq_lengths.data.cpu().numpy(), batch_first=True
                )

                outputs, final_states = self.encoder_rnn(inputs)
            else:
                padded_tokens = padded_tokens[:, 0 : lengths[0]]
                outputs, final_states = self.encoder_rnn(self.input_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.encoder_type == self.ATT_GRU else final_states

    def forward(self, inputs, attention_memory=None):
        """inputs: [n_batch input strings].
        attention_memory: hidden state from recurrent encoder to drive attention.
        outputs: [n_batch x embedding_dim] tensor.
        """
        padded_tokens, token_lengths = self._input_strings_to_padded_token_tensor(
            inputs
        )
        rnn_embeddings = self._padded_token_tensor_to_rnn_embeddings(
            padded_tokens, token_lengths
        )

        if self.encoder_type == self.ATT_GRU:
            # outputs: B x L x D
            # memory: B x M
            mask = (padded_tokens != 0).float()
            # From BabyAI comments (obs.instr = batch of language)
            # The mask tensor has the same length as the padded tokens, and thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, : rnn_embeddings.shape[1]]
            rnn_embeddings = rnn_embeddings[:, : mask.shape[1]]

            keys = self.att2key(attention_memory)
            pre_softmax = (keys[:, None, :] * rnn_embeddings).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            rnn_embeddings = (rnn_embeddings * attention[:, :, None]).sum(1)
        return rnn_embeddings


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SingleImageExampleEncoder(nn.Module):
    """Image encoder for tasks specified by a single image example.  Reference implementation: LOGO Feature CNN from DreamCoder."""

    ENCODER_INPUT_DIM = 128  # Expects 128 x 128 image pixels
    ENCODER_HIDDEN_DIM = 64
    ENCODER_INPUT_CHANNELS = 1
    ENCODER_KERNEL_SIZE = 3
    PIXEL_NORMALIZATION_CONSTANT = 256.0

    def __init__(
        self,
        experiment_state,
        encoder_input_dim=ENCODER_INPUT_DIM,
        encoder_input_channels=ENCODER_INPUT_CHANNELS,
        encoder_kernel_size=ENCODER_KERNEL_SIZE,
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        endpool=True,
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


# SyntaxRobustfill model.
class SyntaxRobustfill(nn.Module):
    """Syntax-aware Robustfill model."""

    pass
