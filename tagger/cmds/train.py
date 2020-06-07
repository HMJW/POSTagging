# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from tagger import Tagger, Model
from tagger.metric import SpanF1Method
from tagger.utils import Corpus, Embedding, Vocab
from tagger.utils.data import TextDataset, batchify

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ftrain', default='data/PTB/train.txt',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/PTB/dev.txt',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/PTB/test.txt',
                               help='path to test file')
        subparser.add_argument('--fembed', default='../data/embedding/glove.6B.100d.txt',
                               help='path to pretrained embeddings')
        subparser.add_argument('--unk', default="unk",
                               help='unk token in pretrained embeddings')

        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)
        if config.preprocess or not os.path.exists(config.vocab):
            vocab = Vocab.from_corpus(corpus=train, min_freq=1)
            vocab.collect(corpus=train+dev+test, min_freq=1)
            vocab.read_embeddings(Embedding.load(config.fembed, config.unk))
            torch.save(vocab, config.vocab)
        else:
            vocab = torch.load(config.vocab)
        config.update({
            'n_words': vocab.n_init,
            'n_chars': vocab.n_chars,
            'n_labels': vocab.n_labels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        print(vocab)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))
        # set the data loaders
        train_loader = batchify(trainset, config.batch_size, False)
        dev_loader = batchify(devset, config.batch_size)
        test_loader = batchify(testset, config.batch_size)
        print(f"{'train:':6} {len(trainset):5} sentences, {train.nwords} words in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences, {dev.nwords} words in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences, {test.nwords} words in total, "
              f"{len(test_loader):3} batches provided")

        print("Create the model")
        tagger = Tagger(config, vocab.embed).to(config.device)
        print(f"{tagger}\n")

        optimizer = Adam(tagger.parameters(), config.lr)
        model = Model(config, vocab, tagger, optimizer)

        total_time = timedelta()
        best_e, best_metric = 1, SpanF1Method(vocab)

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(train_loader)

            print(f"Epoch {epoch} / {config.epochs}:")
            loss, train_metric = model.evaluate(train_loader)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = model.evaluate(dev_loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = model.evaluate(test_loader)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                model.tagger.save(config.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.tagger = Tagger.load(config.model)
        loss, metric = model.evaluate(test_loader)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
