# -*- coding: utf-8 -*-

from tagger import Tagger, Model
from tagger.utils import Corpus
from tagger.utils.data import TextDataset, batchify

import torch


class Predict(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--batch-size', default=64, type=int,
                               help='batch size')
        subparser.add_argument('--fdata', default='data/PTB/test.txt',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.bmes',
                               help='path to predicted result')
        return subparser

    def __call__(self, config):
        print("Load the model")
        vocab = torch.load(config.vocab)
        tagger = Tagger.load(config.model)
        model = Model(config, vocab, tagger)

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus, False))
        # set the data loader
        loader = batchify(dataset, config.batch_size)

        print("Make predictions on the dataset")
        corpus.labels = model.predict(loader)

        print(f"Save the predicted result to {config.fpred}")
        corpus.save(config.fpred)
