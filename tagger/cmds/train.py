# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from tagger import Tagger, Model
from tagger.metric import ManyToOneAccuracy, AccuracyMethod
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
        subparser.add_argument('--ftrain', default='data/PTB/train.tsv',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/PTB/dev.tsv',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/PTB/test.tsv',
                               help='path to test file')
        subparser.add_argument('--use_dict', action="store_true", default=False,
                        help='path to test file')

        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)
        train = train + dev + test
        if config.preprocess or not os.path.exists(config.vocab):
            vocab = Vocab.from_corpus(corpus=train, min_freq=1)
            vocab.collect(corpus=train, min_freq=1)
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

        # set the data loaders
        train_loader = batchify(trainset, config.batch_size, False)
        print(f"{'train:':6} {len(trainset):5} sentences, {train.nwords} words in total, "
              f"{len(train_loader):3} batches provided")

        print("Create the model")
        tagger = Tagger(config)
        if config.use_dict:
            tagger.reset_parameters(vocab)
        tagger = tagger.to(config.device)
        print(f"{tagger}\n")
        model = Model(config, vocab, tagger)

        total_time = timedelta()
        best_e, best_metric = 1, ManyToOneAccuracy(vocab.n_labels)
        last_loss, count = 0, 0

        loss, acc_metric, manytoOne_metric = model.evaluate(train_loader)
        print(f"{'train:':6} Loss: {loss:.4f} {manytoOne_metric} {acc_metric}")

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(train_loader)

            print(f"Epoch {epoch} / {config.epochs}:")
            loss, train_metric, manytoOne_metric = model.evaluate(train_loader)
            print(f"{'train:':6} Loss: {loss:.4f} {manytoOne_metric} {train_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if epoch > 1 and abs(last_loss - loss) <= 2e-5:
                count += 1
            else:
                count = 0
            last_loss = loss
            if manytoOne_metric > best_metric:
                best_e, best_metric = epoch, manytoOne_metric
                model.tagger.save(config.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if count >= config.patience:
                break
        model.tagger = Tagger.load(config.model)
        loss, _, many2one = model.evaluate(train_loader)

        print(f"max score of test is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {many2one.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
