# -*- coding: utf-8 -*-

from tagger.modules import CHAR_LSTM, CRF

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Tagger(nn.Module):

    def __init__(self, config, embed):
        super(Tagger, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embed)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_chars=config.n_chars,
                                   n_embed=config.n_char_embed,
                                   n_out=config.n_char_out)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)

        # the word-lstm layer
        self.lstm = nn.LSTM(input_size=config.n_embed + config.n_char_out,
                            hidden_size=config.n_lstm_hidden,
                            batch_first=True,
                            bidirectional=True)

        # the MLP layers
        self.hid = nn.Linear(config.n_lstm_hidden * 2, config.n_lstm_hidden)
        self.activation = nn.Tanh()
        self.out = nn.Linear(config.n_lstm_hidden, config.n_labels)

        # CRF layer
        self.crf = CRF(config.n_labels)

        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        # init Linear
        nn.init.xavier_uniform_(self.hid.weight)
        nn.init.xavier_uniform_(self.out.weight)
        # init word emb
        nn.init.zeros_(self.word_embed.weight)

    def forward(self, words, chars):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.pretrained(words) + self.word_embed(ext_words)
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)

        # concatenate the word and char representations
        x = torch.cat((word_embed, char_embed), dim=-1)
        x = self.embed_dropout(x)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = x[inverse_indices]

        x = self.hid(x)
        x = self.activation(x)
        x = self.out(x)

        return x

    @classmethod
    def load(cls, fname):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embed'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embed': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)
