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


    def train(self, loader):
        self.tagger.train()

        for words, chars, labels, possible_labels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            s_emit = self.tagger(words)
            margial = self.tagger.get_logZ(s_emit, mask)
            loss = -margial
            loss.backward()
            nn.utils.clip_grad_value_(self.tagger.parameters(),
                                     self.config.clip)
            self.optimizer.step()
            

    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric = 0, AccuracyMethod()
        
        for words, chars, labels, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            margial = self.tagger.get_logZ(s_emit, mask)
            loss += -margial * words.size(0)

            predicts = self.tagger.viterbi(s_emit, mask)
            metric(predicts, targets)

        loss /= len(loader.dataset)

        return float(loss), metric

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
            predicts = self.tagger.crf.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

