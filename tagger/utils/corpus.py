# -*- coding: utf-8 -*-

from collections import namedtuple


Sentence = namedtuple(typename='Sentence',
                      field_names=['word', 'label'],
                      defaults=[None]*2)


class Corpus(object):

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i))
                      for i in zip(*(f for f in sentence if f))) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return [list(sentence.word) for sentence in self]

    @property
    def labels(self):
        return [list(sentence.label) for sentence in self]

    @labels.setter
    def labels(self, sequences):
        self.sentences = [sentence._replace(label=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line.strip() for line in f]
        for i, line in enumerate(lines):
            if not line:
                sentence = Sentence(*zip(*[l.split() for l in lines[start:i]]))
                sentences.append(sentence)
                start = i + 1
        corpus = cls(sentences)

        return corpus

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")
