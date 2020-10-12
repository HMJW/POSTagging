# -*- coding: utf-8 -*-

import unicodedata
from collections import Counter, defaultdict
from torch.nn.utils.rnn import pad_sequence
import re

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
        s += f"{sum(label)/len(label):.2f} avg labels in dict, "
        s += f"{len(self.templates)} n_features, "
        s += f"{len(self.tri_grams)} n_trigrams"
        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word, self.unk_index)
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
        words = [word for word in embed.tokens]
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
        tri_grams = tri_grams | set(word)

        # 2-gram
        for i in range(1, len(word)):
            tri_grams.add(word[i-1:i+1])

        # 3-gram
        for i in range(2, len(word)):
            tri_grams.add(word[i-2:i+1])
        
        return template, tri_grams

    def templates2id(self, templates):
        ids = [self.templates.get(t, self.pad_index) for t in templates]
        return torch.tensor(ids)

    def trigrams2id(self, trigrams):
        ids = [self.tri_grams.get(gram, self.pad_index) for gram in trigrams] 
        return torch.tensor(ids)

    def create_feature_space(self, corpus):
        templates = set()
        tri_grams = set()
        for seq in corpus.words:
            for word in seq:
                template, tri_gram = self.get_feature_template(word)
                for t in template:
                    templates.add(t)
                for gram in tri_gram:
                    tri_grams.add(gram)
        templates = [self.pad, self.unk] + sorted(templates)
        tri_grams = [self.pad, self.unk] + sorted(tri_grams)
        self.templates = {t:i for i, t in enumerate(templates)}
        self.tri_grams = {t:i for i, t in enumerate(tri_grams)}

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        labels = list({label for seq in corpus.labels for label in seq})
        vocab = cls(words, chars, labels)

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



