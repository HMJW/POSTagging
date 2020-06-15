from tagger.utils import Corpus
from tagger.utils import Corpus, Embedding, Vocab
from tagger.utils.data import TextDataset, batchify
from torch.nn.utils.rnn import pad_sequence
import torch
from hmmlearn import hmm
import mkl
mkl.set_num_threads(8)


train = Corpus.load("data/PTB/train.tsv")
dev = Corpus.load("data/PTB/dev.tsv")
test = Corpus.load("data/PTB/test.tsv")
train = train + dev + test
print(f"{'train:':6} {len(train):5} sentences, {train.nwords} words in total, ")

vocab = Vocab.from_corpus(corpus=train, min_freq=1)
vocab.collect(corpus=train, min_freq=1)

trainset = TextDataset(vocab.numericalize(train))
lengths = [len(x) for x in trainset.items[0]]
X = torch.cat(trainset.items[0], dim=-1)
X = X.view(-1, 1)
X = X.numpy()

model = hmm.MultinomialHMM(n_components=vocab.n_labels, verbose=True, n_iter=1000)

model.fit(X, lengths)