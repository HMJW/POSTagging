# -*- coding: utf-8 -*-

import unicodedata
from collections import Counter, defaultdict
from torch.nn.utils.rnn import pad_sequence
import re

import torch


class Vocab(object):

    def __init__(self, words, labels):
        self.words = sorted(words)
        self.labels = sorted(labels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.label_dict = {label: i for i, label in enumerate(self.labels)}

        self.n_words = len(self.words)
        self.n_labels = len(self.labels)
        self.n_init = self.n_words

    def __repr__(self):
        s = f"{self.__class__.__name__}: "
        s += f"{self.n_words} n_words, "
        s += f"{self.n_labels} n_labels, "
        s += f"{len(self.possible_dict)} n_possible_dict, "
        label = [len(y) for _, y in self.possible_dict.items()]
        s += f"{sum(label)/len(label):.2f} avg labels in dict, "
        s += f"{len(self.features)} n_features, "
        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word, 0)
                             for word in sequence])


    def label2id(self, sequence):
        return torch.tensor([self.label_dict.get(label, 0)
                             for label in sequence])

    def id2label(self, ids):
        return [self.labels[i] for i in ids]

    def numericalize(self, corpus, training=True):
        words = [self.word2id(seq) for seq in corpus.words]
        if not training:
            return (words, )
        labels = [self.label2id(seq) for seq in corpus.labels]

        return words, labels

    def get_feature_template(self, word):
        template = set()
        
        template.add(word)

        if bool(re.search(r'\d', word)):
            template.add("<containsDigit>")

        if bool(re.search(r'-', word)):
            template.add("<containsHyphen>")
        
        if word.istitle():
            template.add("<Cap>")
        
        # 1-gram
        template = template | set(word)

        # 2-gram
        for i in range(1, len(word)):
            template.add(word[i-1:i+1])

        # 3-gram
        for i in range(2, len(word)):
            template.add(word[i-2:i+1])
        
        return template

    def templates2id(self, templates):
        ids = [self.features.get(t, 0) for t in templates]
        return torch.tensor(ids)


    def create_feature_space(self, corpus):
        self.features = {}
        for seq in corpus.words:
            for word in seq:
                template = self.get_feature_template(word)
                for t in template:
                    if t not in self.features:
                        self.features[t] = len(self.features)

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        labels = list({label for seq in corpus.labels for label in seq})
        vocab = cls(words, labels)

        return vocab
    
    def collect(self, corpus, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = set(word for word, freq in words.items() if freq >= min_freq)
        d = defaultdict(set)
        for seq in corpus.sentences:
            for word, label in zip(seq.word, seq.label):
                if word in words:
                    d[word].add(label)
        self.possible_dict = dict(d)

    
    def get_all_words_features(self):
        templates = []
        for word in self.words:
            template = self.get_feature_template(word)
            templates.append(self.templates2id(template))
        self.all_words_features = templates



