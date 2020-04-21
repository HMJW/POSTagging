# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import Corpus
from parser.utils.data import TextDataset, batchify

import torch


class Evaluate(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--fdata', default='../data/treebanks/codt/test.conll',
                               help='path to dataset')
        subparser.add_argument('--tree', action='store_true',
                               help='whether to force tree')
        return subparser

    def __call__(self, config):
        print("Load the model")
        vocab = torch.load(config.vocab)
        parser = BiaffineParser.load(config.model)
        model = Model(config, vocab, parser)

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus), config.buckets)
        # set the data loader
        loader = batchify(dataset, config.batch_size)

        print("Evaluate the dataset")
        loss, metric = model.evaluate(loader, config.punct)
        print(f"Loss: {loss:.4f} {metric}")
