from tagger.utils import Corpus
from tagger.utils import Corpus, Embedding, Vocab
import random

test = Corpus.load("data/PTB/test.tsv")
# test = Corpus.load("pred.tsv")

print(f"{'test:':6} {len(test):5} sentences, {test.nwords} words in total, ")

nbigram = set()
for sentence in test.sentences:
    for i in range(len(sentence.label) - 1):
        nbigram.add((sentence.label[i], sentence.label[i+1]))
print(len(nbigram))