# -*- coding: utf-8 -*-

from tagger.metric import SpanF1Method
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

        for words, chars, labels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            s_emit = self.tagger(words, chars)
            loss = self.tagger.crf(s_emit, labels, mask)

            loss.backward()
            nn.utils.clip_grad_norm_(self.tagger.parameters(),
                                     self.config.clip)
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric = 0, SpanF1Method(self.vocab)

        for words, chars, labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words, chars)
            loss += self.tagger.crf(s_emit, labels, mask)
            predicts = self.tagger.crf.viterbi(s_emit, mask)

            metric(predicts, targets)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_labels = []
        for words, chars in loader:
            mask = words.ne(self.vocab.pad_index)

            s_emit = self.tagger(words, chars)
            predicts = self.tagger.crf.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

