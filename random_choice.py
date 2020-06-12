from tagger.utils import Corpus
from tagger.utils import Corpus, Embedding, Vocab
import random


train = Corpus.load("data/PTB/train.tsv")
dev = Corpus.load("data/PTB/dev.tsv")
test = Corpus.load("data/PTB/test.tsv")
train = train + dev + test
print(f"{'train:':6} {len(train):5} sentences, {train.nwords} words in total, ")
# print(f"{'dev:':6} {len(dev):5} sentences, {dev.nwords} words in total, ")
# print(f"{'test:':6} {len(test):5} sentences, {test.nwords} words in total, ")

vocab = Vocab.from_corpus(corpus=train, min_freq=1)
vocab.collect(corpus=train, min_freq=1)

total, correct = 0, 0
for sentence in train.sentences[:1000]:
    pred = [random.choice(list(vocab.possible_dict[word.lower()])) for word in sentence.word]
    total += len(pred)
    for x, y in zip(pred, sentence.label):
        if x == y:
            correct += 1
print(correct / total)

