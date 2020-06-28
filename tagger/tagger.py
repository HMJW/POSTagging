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
        emits = torch.ones(self.config.n_labels, self.config.n_words)
        strans = torch.ones(self.config.n_labels)
        etrans = torch.ones(self.config.n_labels)

        nn.init.normal_(trans, 0, 3)
        nn.init.normal_(emits, 0, 3)
        nn.init.normal_(strans, 0, 3)
        nn.init.normal_(etrans, 0, 3)

        self.trans = nn.Parameter(trans)
        self.strans = nn.Parameter(strans)
        self.emits = nn.Parameter(emits)
        self.etrans = nn.Parameter(etrans)


    def extra_repr(self):
        info = f"n_tags={self.n_tags}, n_words={self.n_words}"

        return info

    def reset_parameters(self, vocab):
        trans = torch.ones(self.config.n_labels, self.config.n_labels)
        emits = torch.ones(self.config.n_labels, self.config.n_words)
        strans = torch.ones(self.config.n_labels)
        etrans = torch.ones(self.config.n_labels)

        nn.init.normal_(trans, 0, 3)
        nn.init.normal_(emits, 0, 3)
        nn.init.normal_(strans, 0, 3)
        nn.init.normal_(etrans, 0, 3)

        emits[:, vocab.pad_index] = float("-inf")
        for word, plabels in vocab.possible_dict.items():
            iplabels = set(vocab.labels) - set(plabels)
            index = vocab.label2id(iplabels)
            emits[index, vocab.word_dict[word]] = float("-inf")

        self.trans = nn.Parameter(trans)
        self.strans = nn.Parameter(strans)
        self.emits = nn.Parameter(emits)
        self.etrans = nn.Parameter(etrans)

    def forward(self, words):
        # get the mask and lengths of given batch
        batch_size = words.size(0)
        emits = self.emits - torch.logsumexp(self.emits, dim=-1).unsqueeze(-1)
        x = emits.unsqueeze(0).repeat(batch_size,1,1).gather(-1, words.unsqueeze(1).repeat(1,self.n_tags,1))

        return x.transpose(1, 2)
    
    def get_logZ(self, emit, mask):
        strans = self.strans - torch.logsumexp(self.strans, dim=-1)
        etrans = self.etrans - torch.logsumexp(self.etrans, dim=-1)
        trans = self.trans - torch.logsumexp(self.trans, dim=-1).unsqueeze(-1)

        if self.training:
            strans.register_hook(lambda x: x.masked_fill_(~torch.isfinite(x), 0))
            etrans.register_hook(lambda x: x.masked_fill_(~torch.isfinite(x), 0))
            trans.register_hook(lambda x: x.masked_fill_(~torch.isfinite(x), 0))
            emit.register_hook(lambda x: x.masked_fill_(~torch.isfinite(x), 0))

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
        strans = self.strans - torch.logsumexp(self.strans, dim=-1)
        etrans = self.etrans - torch.logsumexp(self.etrans, dim=-1)
        trans = self.trans - torch.logsumexp(self.trans, dim=-1).unsqueeze(-1)

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
