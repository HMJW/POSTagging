# -*- coding: utf-8 -*-

from tagger.metric import AccuracyMethod
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
        self.criterion = nn.CrossEntropyLoss()


    def train(self, loader):
        self.tagger.train()

        for words, chars, labels, possible_labels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            s_emit = self.tagger(words, chars)
            logZ = self.tagger.crf.get_logZ(s_emit, mask)
            s_emit[~possible_labels] -= 100000

            possible_logZ = self.tagger.crf.get_logZ(s_emit, mask)
            loss = logZ - possible_logZ

            loss.backward()
            nn.utils.clip_grad_norm_(self.tagger.parameters(),
                                     self.config.clip)
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        metric = AccuracyMethod()

        for words, chars, labels, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            s_emit[~possible_labels] = 0
            predicts = self.tagger.viterbi(s_emit, mask)
            metric(predicts, targets)

        return metric

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_labels = []
        for words, chars, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)

            s_emit = self.tagger(words, chars)
            s_emit[~possible_labels] -= 100000
            predicts = self.tagger.crf.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

