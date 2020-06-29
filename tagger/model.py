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

    def pad_left(self, tensor, lens):
        res = tensor.new_zeros(tensor.shape)
        for i, length in enumerate(lens):
            res[i, :length] = tensor[i, -length:]
        return res

    @torch.no_grad()
    def train(self, loader):
        strans_grad = torch.full([self.tagger.n_tags], 0)
        etrans_grad = torch.full([self.tagger.n_tags], 0)
        weight_grad = torch.full([self.tagger.n_features, self.tagger.n_tags], 0)
        trans_grad = torch.full([self.tagger.n_tags, self.tagger.n_tags], 0)

        emits = self.tagger.get_emits(self.vocab)
        strans = self.tagger.strans - torch.logsumexp(self.tagger.strans, dim=-1)
        etrans = self.tagger.etrans - torch.logsumexp(self.tagger.etrans, dim=-1)
        trans = self.tagger.trans - torch.logsumexp(self.tagger.trans, dim=-1).unsqueeze(-1)

        feature_partial_Z = torch.full([self.tagger.n_features, self.tagger.n_tags], 0)
        for i in range(self.vocab.n_words):
            features = self.vocab.all_words_features[i]
            p = torch.exp(emits[i])
            feature_partial_Z[features] += p


        for words, _ in loader:
            mask = words.ge(0).unsqueeze(0)
            length = len(words)
            s_emit = self.tagger(words, emits)
            s_emit = s_emit.unsqueeze(0)

            # T*N
            alpha = self.tagger.forw(s_emit, mask)[0]
            beta = self.tagger.back(s_emit, mask)[0]
            logZ = torch.logsumexp(alpha[length-1,:] + etrans, dim=0)
            assert torch.isfinite(logZ).all()

            gamma = alpha + beta
            posteriors = gamma - logZ

            strans_grad = strans_grad + torch.exp(posteriors[0])
            etrans_grad = etrans_grad + torch.exp(posteriors[-1])

            s_emit = s_emit[0]
            for t in range(length):
                p = torch.exp(posteriors[t])
                features = self.vocab.all_words_features[words[t]]
                grad = p.new_zeros([self.tagger.n_features, self.tagger.n_tags])
                grad[features] += 1
                grad = (grad - feature_partial_Z) * p
                weight_grad += grad

            # N*1 + 1*N + N*N + 1*N
            if length > 1:
                log_xi_sum = [alpha[l].unsqueeze(-1) + beta[l+1].unsqueeze(0) + trans + s_emit[l+1].unsqueeze(0) for l in range(length-1)]
                log_xi_sum = torch.stack(log_xi_sum, 0)
                log_xi_sum = torch.logsumexp(log_xi_sum, dim=0) - logZ
                trans_grad = trans_grad + torch.exp(log_xi_sum)

        strans_grad = strans_grad * (1 - torch.exp(strans))
        etrans_grad = etrans_grad * (1 - torch.exp(etrans))
        trans_grad = trans_grad * (1 - torch.exp(trans))

        self.tagger.strans.data += strans_grad * 0.1 - self.tagger.strans.data * 0
        self.tagger.etrans.data += etrans_grad * 0.1 - self.tagger.etrans.data * 0
        self.tagger.trans.data += trans_grad * 0.1 - self.tagger.trans.data * 0
        self.tagger.weights.data += weight_grad * 0.1 - self.tagger.weights.data * 0

    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric, manyToOne = 0, AccuracyMethod(), ManyToOneAccuracy(self.vocab.n_labels)
        
        emits = self.tagger.get_emits(self.vocab)
        for words, labels in loader:
            length = len(words)

            s_emit = self.tagger(words, emits)
            s_emit, mask = s_emit.unsqueeze(0), words.ge(0).unsqueeze(0)
            margial = self.tagger.get_logZ(s_emit, mask)
            loss += -margial

            predicts = self.tagger.viterbi(s_emit, mask)
            labels = labels.unsqueeze(0)
            metric(predicts, labels)
            manyToOne(predicts, labels)

        loss /= len(loader.dataset)

        return float(loss), metric, manyToOne

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_labels = []
        for words in loader:
            words = next(words)
            mask = words.ne(self.vocab.pad_index)
            s_emit = self.tagger(words)
            predicts = self.tagger.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

