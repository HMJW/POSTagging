# -*- coding: utf-8 -*-

from tagger.metric import AccuracyMethod, ManyToOneAccuracy
import torch
import torch.nn as nn


class Model(object):

    def __init__(self, config, vocab, tagger, optimizer=None, scheduler=None):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.tagger = tagger
        self.optimizer = optimizer
        self.scheduler = scheduler

    @torch.no_grad()
    def train(self, loader):
        self.tagger.train()

        strans_numerator = []
        strans_denominator = []
        etrans_numerator = []
        etrans_denominator = []

        emits_numerator = []
        emits_denominator = []

        trans_numerator = []
        trans_denominator = []

        for words, chars, labels, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            batch_size, lens = mask.size(0), mask.sum(1)
            s_emit = self.tagger(words)

            alpha = self.tagger.forw(s_emit, mask)
            beta = self.tagger.back(s_emit, mask)
            
            for i, length in enumerate(lens):
                forward, backward = alpha[i, :length], beta[i, -length:]
                logZ = torch.logsumexp(forward[-1] + torch.log(self.tagger.etrans), dim=-1)
                logZ2 = torch.logsumexp(backward[0] + torch.log(self.tagger.strans) + torch.log(s_emit[i, 0]), dim=-1)
                assert torch.isfinite(logZ)
                gamma = forward + backward
                
                # update strans
                strans_numerator.append(gamma[0])
                strans_denominator.append(logZ)

                # update etrans
                etrans_numerator.append(gamma[-1])
                etrans_denominator.append(logZ)

                # update emit
                logp_emit = (forward + backward) - logZ
                emits_denominator.append(torch.logsumexp(logp_emit, dim = 0))
                count = torch.arange(self.tagger.n_words).unsqueeze(-1).to(words.device) == words[i,:length]
                count = torch.log(count.float() @ torch.exp(logp_emit))
                count = count.transpose(0, 1)
                emits_numerator.append(count)
                
                # update trans
                if length > 1:
                    marginal = [forward[l].unsqueeze(-1) + backward[l+1].unsqueeze(0) + torch.log(self.tagger.trans) + torch.log(self.tagger.emits[:, words[i, l+1]]) for l in range(length-1)]
                    marginal = torch.stack(marginal, 0)
                    marginal = torch.logsumexp(marginal, dim=0) - logZ
                    trans_numerator.append(marginal)
                    logp_emit_t = torch.logsumexp(((forward + backward) - logZ)[:-1], dim=0)
                    trans_denominator.append(logp_emit_t)

            strans_numerator = [torch.logsumexp(torch.stack(strans_numerator), 0)]
            strans_denominator = [torch.logsumexp(torch.stack(strans_denominator,-1).squeeze(), -1)]

            etrans_numerator = [torch.logsumexp(torch.stack(etrans_numerator), 0)]
            etrans_denominator = [torch.logsumexp(torch.stack(etrans_denominator,-1).squeeze(), -1)]
            
            emits_numerator = [torch.logsumexp(torch.stack(emits_numerator, 0), 0)]
            emits_denominator = [torch.logsumexp(torch.stack(emits_denominator), 0)]

            if len(trans_denominator) > 0:
                trans_numerator = [torch.logsumexp(torch.stack(trans_numerator, 0), 0)]
                trans_denominator = [torch.logsumexp(torch.stack(trans_denominator), 0)]

        strans_numerator = strans_numerator[0]
        strans_denominator = strans_denominator[0]
        etrans_numerator = etrans_numerator[0]
        etrans_denominator = etrans_denominator[0]

        emits_numerator = emits_numerator[0]
        emits_denominator = emits_denominator[0]

        trans_numerator = trans_numerator[0]
        trans_denominator = trans_denominator[0]

        to_keep_strans = ~torch.isfinite(strans_numerator)
        self.tagger.strans.data = self.tagger.strans.data * to_keep_strans + torch.exp(strans_numerator - strans_denominator) + 1e-6
        self.tagger.strans.data = self.tagger.strans.data / self.tagger.strans.data.sum()

        to_keep_etrans = ~torch.isfinite(etrans_numerator)
        self.tagger.etrans.data = self.tagger.etrans.data * to_keep_etrans + torch.exp(etrans_numerator- etrans_denominator) + 1e-6
        self.tagger.etrans.data = self.tagger.etrans.data / self.tagger.etrans.data.sum()

        column_to_keep_emits = ~torch.isfinite(emits_numerator) 
        rows_to_keep_emits = ~torch.isfinite(emits_denominator)
        emits_denominator[rows_to_keep_emits] = 0
        to_keep_emits = column_to_keep_emits + rows_to_keep_emits.unsqueeze(-1)
        self.tagger.emits.data = self.tagger.emits.data * to_keep_emits + torch.exp(emits_numerator - emits_denominator.unsqueeze(-1)) + 1e-6
        self.tagger.emits.data = self.tagger.emits.data / self.tagger.emits.data.sum(-1).unsqueeze(-1)

        rows_to_keep_trans = ~torch.isfinite(trans_denominator)
        trans_denominator[rows_to_keep_trans] = 0
        self.tagger.trans.data = self.tagger.trans.data * rows_to_keep_trans + torch.exp(trans_numerator - trans_denominator) + 1e-6
        self.tagger.trans.data = self.tagger.trans.data / self.tagger.trans.data.sum(-1).unsqueeze(-1)


    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric, manyToOne = 0, AccuracyMethod(), ManyToOneAccuracy(self.vocab.n_labels)
        
        for words, chars, labels, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            margial = self.tagger.get_logZ(s_emit, mask)
            loss += -margial * words.size(0)

            predicts = self.tagger.viterbi(s_emit, mask)
            metric(predicts, targets)
            manyToOne(predicts, targets)

        loss /= len(loader.dataset)

        return float(loss), metric, manyToOne

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_labels = []
        for words, chars, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            s_emit[~possible_labels] = 0
            predicts = self.tagger.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

