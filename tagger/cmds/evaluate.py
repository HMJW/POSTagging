# -*- coding: utf-8 -*-

from tagger import Tagger, Model
from tagger.utils import Corpus
from tagger.utils.data import TextDataset, batchify

import torch


class Evaluate(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--batch-size', default=64, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/PTB/test.tsv',
                               help='path to dataset')
        return subparser

    def __call__(self, config):
        print("Load the model")
        vocab = torch.load(config.vocab)
        tagger = Tagger.load(config.model)
        model = Model(config, vocab, tagger)

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus))
        # set the data loader
        loader = batchify(dataset, config.batch_size)

        print("Evaluate the dataset")
        loss, acc_metric, many2one_metric = model.evaluate(loader,)
        print(f"{'train:':6} Loss: {loss:.4f} {manytoOne_metric} {acc_metric}")
