# -*- coding: utf-8 -*-

from parser.metric import Metric
from parser.utils.alg import eisner
import torch
import torch.nn as nn


class Model(object):

    def __init__(self, config, vocab, parser):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.parser = parser
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader):
        self.parser.train()

        for words, chars, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            partial_mask = arcs.ne(-1)
            mask = mask & partial_mask
            s_arc, s_rel = self.parser(words, chars)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parser.parameters(),
                                     self.config.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader, punct=False):
        self.parser.eval()

        loss, metric = 0, Metric()

        for words, chars, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            s_arc, s_rel = self.parser(words, chars)
            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask)

            partial_mask = arcs.ne(-1)
            mask = mask & partial_mask
            # ignore all punctuation if not specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)

            gold_arcs, gold_rels = arcs[mask], rels[mask]
            loss += self.get_loss(s_arc[mask], s_rel[mask], gold_arcs, gold_rels)
            metric(pred_arcs[mask], pred_rels[mask], gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_arcs, all_rels, all_probs = [], [], []
        for words, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_arc, s_rel = self.parser(words, chars)
            p_arc, p_rel = s_arc.softmax(dim=-1), s_rel.softmax(dim=-1)

            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask)
            pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
            p_arc, p_rel = p_arc[mask], p_rel[mask]
            pred_p_arcs = p_arc[torch.arange(len(p_arc)), pred_arcs] 
            pred_p_rels = p_rel[torch.arange(len(p_rel)), pred_arcs, pred_rels]

            pred_probs = pred_p_arcs * pred_p_rels
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
            all_probs.extend(torch.split(pred_probs, lens))
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]
        all_probs = [seq.tolist() for seq in all_probs]

        return all_arcs, all_rels, all_probs

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel, mask):
        if self.config.tree:
            arc_preds = eisner(s_arc, mask)
        else:
            arc_preds = s_arc.argmax(-1)
        rel_preds = s_rel.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds
