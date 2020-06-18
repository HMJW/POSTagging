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


    def train(self, loader):
        self.tagger.train()
        self.optimizer.zero_grad()
        for words, _ in loader:
            mask = words.ne(self.vocab.pad_index)
            s_emit = self.tagger(words)
            likelyhood = self.tagger.get_logZ(s_emit, mask)
            loss = -likelyhood
            loss.backward()
        self.optimizer.step()
            

    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric, many2one_metric = 0, AccuracyMethod(), ManyToOneAccuracy(self.vocab.n_labels)
        
        for words, labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            likelyhood = self.tagger.get_logZ(s_emit, mask)
            loss += -likelyhood

            predicts = self.tagger.viterbi(s_emit, mask)
            metric(predicts, targets)
            many2one_metric(predicts, targets)

        loss /= len(loader.dataset)

        return float(loss), metric, many2one_metric

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

