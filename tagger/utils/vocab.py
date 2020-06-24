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
        s += f"{len(self.tri_grams)} n_trigrams"
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
        template, tri_grams = [], set()
        
        template.append(word)

        if bool(re.search(r'\d', word)):
            template.append("<containsDigit>")

        if bool(re.search(r'-', word)):
            template.append("<containsHyphen>")
        
        if word.istitle():
            template.append("<Cap>")
        
        # 1-gram
        tri_grams = tri_grams & set(word)

        # 2-gram
        for i in range(1, len(word)):
            tri_grams.add(word[i-1:i+1])

        # 3-gram
        for i in range(2, len(word)):
            tri_grams.add(word[i-2:i+1])
        
        return template, tri_grams

    def templates2id(self, templates):
        ids = [self.features.get(t, 0) for t in templates]
        return torch.tensor(ids)

    def trigrams2id(self, trigrams):
        ids = [self.tri_grams.get(gram, 0) for gram in trigrams] 
        return torch.tensor(ids)

    def create_feature_space(self, corpus):
        self.features = {}
        self.tri_grams = {}
        for seq in corpus.words:
            for word in seq:
                template, tri_grams = self.get_feature_template(word)
                for t in template:
                    if t not in self.features:
                        self.features[t] = len(self.features)
                for gram in tri_grams:
                    if gram not in self.tri_grams:
                        self.tri_grams[gram] = len(self.tri_grams)

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
        templates, trigrams = [], []
        for word in self.words:
            template, trigram = self.get_feature_template(word)
            templates.append(self.templates2id(template))
            trigrams.append(self.trigrams2id(trigram))
        self.all_words_features = templates
        self.all_words_trigrams = trigrams



