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

        trans = torch.ones(self.config.n_labels, self.config.n_labels)
        weights = torch.zeros(self.config.n_features, self.config.n_labels)
        trigram_weights = torch.zeros(self.config.n_trigrams)
        strans = torch.ones(self.config.n_labels)
        etrans = torch.ones(self.config.n_labels)

        nn.init.uniform_(trans, a=0, b=5)
        nn.init.uniform_(strans, a=0, b=5)
        nn.init.uniform_(etrans, a=0, b=5)

        strans = torch.log(strans.softmax(dim=-1))
        etrans = torch.log(etrans.softmax(dim=-1))
        trans = torch.log(trans.softmax(dim=-1))

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

    def forward(self, templates, trigrams):
        # get the mask and lengths of given batch
        batch_size, max_len = templates.size(0), templates.size(1)

        # feature score
        x = self.weights.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1, 1).gather(2, templates.unsqueeze(-1).repeat(1, 1, 1, self.n_tags))
        x = x.sum(2)

        # trigram score
        y = self.trigram_weights.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1).gather(2, trigrams)
        y = y.sum(-1).unsqueeze(-1)
        return x + y
    
    def forw(self, emit, mask):
        lens = mask.sum(dim=1)
        emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape
        alpha = emit.new_zeros(T, B, N)

        alpha[0] = self.strans + emit[0]  # [B, N]
        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            mask_i = mask[i].unsqueeze(1).expand_as(alpha[i-1])  # [B, N]
            scores = trans_i + emit_i + alpha[i-1].unsqueeze(2)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=1)  # [B, N]
            alpha[i][mask_i] = scores[mask_i]


        return alpha.transpose(0, 1)

    def back(self, emit, mask):
        def pad_right(tensor, lens):
            res = tensor.new_zeros(tensor.shape)
            for i, length in enumerate(lens):
                res[i, -length:] = tensor[i, :length]
            return res
        lens = mask.sum(dim=1)
        emit, mask = pad_right(emit, lens), pad_right(mask, lens)
        emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape    
        beta = emit.new_zeros(T, B, N)

        beta[-1] = self.etrans  # [B, N]
        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[-i]  # [B, N]
            mask_i = mask[-i-1].unsqueeze(1).expand_as(beta[-i])  # [B, N]
            scores = trans_i + (emit_i + beta[-i]).unsqueeze(1)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=2)  # [B, N]
            beta[-i-1][mask_i] = scores[mask_i]


        return beta.transpose(0, 1)


    def get_logZ(self, emit, mask):
        strans = self.strans
        etrans = self.etrans
        trans = self.trans
        
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

        return logZ / B


    def viterbi(self, emit, mask):
        emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape
        lens = mask.sum(dim=0)
        delta = emit.new_zeros(T, B, N)
        paths = emit.new_zeros(T, B, N, dtype=torch.long)

        delta[0] = self.strans + emit[0]  # [B, N]
        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)  # [B, N, N]
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + self.etrans)

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
