# -*- coding: utf-8 -*-

from tagger.modules import CHAR_LSTM, CRF

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Tagger(nn.Module):

    def __init__(self, config):
        super(Tagger, self).__init__()

        self.config = config
        self.n_tags = config.n_labels
        self.n_words = config.n_words
        self.n_features = self.config.n_features
        self.n_trigrams = self.config.n_trigrams
        
        trans = torch.zeros(self.config.n_labels, self.config.n_labels)
        weights = torch.zeros(self.config.n_features, self.config.n_labels)
        trigram_weights = torch.zeros(self.config.n_trigrams)
        strans = torch.zeros(self.config.n_labels)
        etrans = torch.zeros(self.config.n_labels)

        # nn.init.normal_(trans, 0, 5)
        # nn.init.normal_(strans, 0, 5)
        # nn.init.normal_(etrans, 0, 5)
        # nn.init.normal_(weights, 0, 5)
        # nn.init.normal_(trigram_weights, 0, 5)

        self.trans = nn.Parameter(trans)
        self.strans = nn.Parameter(strans)
        self.weights = nn.Parameter(weights)
        self.trigram_weights = nn.Parameter(trigram_weights)
        self.etrans = nn.Parameter(etrans)


    def extra_repr(self):
        info = f"n_tags={self.n_tags}, n_words={self.n_words}"

        return info

    def reset_parameters(self, vocab):
        pass
    
    def get_emits(self, vocab):
        s_features = [self.weights[features].sum(0) for features in vocab.all_words_features]
        s_features = torch.stack(s_features, 0)

        s_trigrams = [self.trigram_weights[trigrams].sum() for trigrams in vocab.all_words_trigrams]
        s_trigrams = torch.stack(s_trigrams, 0)

        emits = s_features + s_trigrams.unsqueeze(-1)
        emits = emits.transpose(0, 1)
        emits = emits - torch.logsumexp(emits, dim=-1).unsqueeze(-1)
        return emits

    def forward(self, words, emits):
        batch_size = words.size(0)
        x = emits.unsqueeze(0).expand(batch_size, -1, -1).gather(-1, words.unsqueeze(1).expand(-1, self.n_tags, -1))
        return x.transpose(1, 2)

    def get_logZ(self, emit, mask):
        strans = self.strans - torch.logsumexp(self.strans, -1)
        etrans = self.etrans - torch.logsumexp(self.etrans, -1)
        trans = self.trans - torch.logsumexp(self.trans, -1).unsqueeze(-1)
        
        emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape

        alpha = strans + emit[0]  # [B, N]
        for i in range(1, T):
            trans_i = trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            mask_i = mask[i].unsqueeze(1).expand_as(alpha)  # [B, N]
            scores = trans_i + emit_i + alpha.unsqueeze(2)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=1)  # [B, N]
            alpha[mask_i] = scores[mask_i]
        logZ = torch.logsumexp(alpha + etrans, dim=1).sum()

        return logZ


    def viterbi(self, emit, mask):
        strans = self.strans - torch.logsumexp(self.strans, -1)
        etrans = self.etrans - torch.logsumexp(self.etrans, -1)
        trans = self.trans - torch.logsumexp(self.trans, -1).unsqueeze(-1)

        emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape
        lens = mask.sum(dim=0)
        delta = emit.new_zeros(T, B, N)
        paths = emit.new_zeros(T, B, N, dtype=torch.long)

        delta[0] = strans + emit[0]  # [B, N]
        for i in range(1, T):
            trans_i = trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)  # [B, N, N]
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + etrans)

            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            # flip the predicted sequence before appending it to the list
            predicts.append(paths.new_tensor(predict).flip(0))

        return predicts

    @classmethod
    def load(cls, fname):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)
