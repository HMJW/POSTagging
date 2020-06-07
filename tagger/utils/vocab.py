# -*- coding: utf-8 -*-

import unicodedata
from collections import Counter, defaultdict

import torch


class Vocab(object):
    pad = '<pad>'
    unk = '<unk>'

    def __init__(self, words, chars, labels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.pad, self.unk] + sorted(words)
        self.chars = [self.pad, self.unk] + sorted(chars)
        self.labels = sorted(labels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.label_dict = {label: i for i, label in enumerate(self.labels)}

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_labels = len(self.labels)
        self.n_init = self.n_words

    def __repr__(self):
        s = f"{self.__class__.__name__}: "
        s += f"{self.n_words} n_words, "
        s += f"{self.n_chars} chars, "
        s += f"{self.n_labels} n_labels, "
        s += f"{len(self.possible_dict)} n_possible_dict, "
        label = [len(y) for _, y in self.possible_dict.items()]
        s += f"{sum(label)/len(label):.2f} avg labels in dict"
        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def label2id(self, sequence):
        return torch.tensor([self.label_dict.get(label, 0)
                             for label in sequence])

    def id2label(self, ids):
        return [self.labels[i] for i in ids]

    def read_embeddings(self, embed, smooth=False):
        words = [word.lower() for word in embed.tokens]
        # if the `unk` token has existed in the pretrained,
        # then replace it with a self-defined one
        if embed.unk:
            words[embed.unk_index] = self.unk

        self.extend(words)
        self.embed = torch.zeros(self.n_words, embed.dim)
        self.embed[self.word2id(words)] = embed.vectors

        if smooth:
            self.embed /= torch.std(self.embed)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus, training=True):
        words = [self.word2id(seq) for seq in corpus.words]
        chars = [self.char2id(seq) for seq in corpus.words]
        possible_labels = []
        for seq in corpus.words:
            x = torch.zeros((len(seq), self.n_labels), dtype=torch.bool)
            for i, word in enumerate(seq):
                word = word.lower()
                if word in self.possible_dict:
                    x[i][self.label2id(self.possible_dict[word])] = 1
                else:
                    x[i] = 1

            possible_labels.append(x)

        if not training:
            return words, chars, possible_labels
        labels = [self.label2id(seq) for seq in corpus.labels]

        return words, chars, labels, possible_labels

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        labels = list({label for seq in corpus.labels for label in seq})
        vocab = cls(words, chars, labels)

        return vocab
    
    def collect(self, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = set(word for word, freq in words.items() if freq >= min_freq)
        d = defaultdict(set)
        for seq in corpus.sentences:
            for word, label in zip(seq.word, seq.label):
                if word.lower() in words:
                    d[word.lower()].add(label)
        self.possible_dict = dict(d)



