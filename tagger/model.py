# -*- coding: utf-8 -*-

from tagger.metric import AccuracyMethod, ManyToOneAccuracy
import torch
import torch.nn as nn


class Model(object):

    def __init__(self, config, vocab, tagger, optimizer=None, scheduler=None):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.tagger = tagger
        self.optimizer = optimizer
        self.scheduler = scheduler

    def pad_left(self, tensor, lens):
        res = tensor.new_zeros(tensor.shape)
        for i, length in enumerate(lens):
            res[i, :length] = tensor[i, -length:]
        return res

    @torch.no_grad()
    def train(self, loader):
        self.tagger.train()
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        strans_numerator = [torch.full([self.tagger.n_tags], float("-inf")).to(device)]
        strans_denominator = [torch.full([], float("-inf")).to(device)]
        etrans_numerator = [torch.full([self.tagger.n_tags], float("-inf")).to(device)]
        etrans_denominator = [torch.full([], float("-inf")).to(device)]

        emits_numerator = [torch.full([self.tagger.n_tags, self.tagger.n_words], float("-inf")).to(device)]
        emits_denominator = [torch.full([self.tagger.n_tags], float("-inf")).to(device)]

        trans_numerator = [torch.full([self.tagger.n_tags, self.tagger.n_tags], float("-inf")).to(device)]
        trans_denominator = [torch.full([self.tagger.n_tags], float("-inf")).to(device)]

        for words, chars, labels, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            batch_size, lens = mask.size(0), mask.sum(1)
            max_len= words.size(1)
            s_emit = self.tagger(words)

            # B*T*N
            alpha = self.tagger.forw(s_emit, mask)
            beta = self.tagger.back(s_emit, mask)
            beta = self.pad_left(beta, lens)
            alpha[~mask] = float("-inf")
            beta[~mask] = float("-inf")

            indexes = torch.arange(batch_size).to(words.device)
            logZs = torch.logsumexp(alpha[indexes, lens-1,:] + self.tagger.etrans, dim=1)

            assert torch.isfinite(logZs).all()
            gamma = alpha + beta

            strans_numerator.append(torch.logsumexp(gamma[indexes, 0], dim=0))
            strans_denominator.append(torch.logsumexp(logZs, 0))

            etrans_numerator.append(torch.logsumexp(gamma[indexes, lens-1], dim=0))
            etrans_denominator.append(torch.logsumexp(logZs, 0))

            logp_emit = (alpha + beta) - logZs.unsqueeze(-1).unsqueeze(-1)
            logp_emit[~mask] = float('-inf')
            emits_denominator.append(torch.logsumexp(torch.logsumexp(logp_emit, dim=1), dim=0))

            count = torch.arange(self.tagger.n_words).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1).to(words.device) == words.unsqueeze(-1)
            count[~mask] = 0
            count = torch.log(torch.exp(logp_emit.transpose(1,2)) @ count.float())
            emits_numerator.append(torch.logsumexp(count, 0))

            # B*N*1 + B*1*N + B*N*N + B*1*N
            if (lens > 1).any():
                marginal = [alpha[:, l].unsqueeze(-1) + beta[:, l+1].unsqueeze(1) + self.tagger.trans.unsqueeze(0) + s_emit[:,l+1].unsqueeze(1) for l in range(max_len-1)]
                marginal = torch.stack(marginal, 0)
                marginal = torch.logsumexp(marginal, dim=0) - logZs.unsqueeze(-1).unsqueeze(-1)
                trans_numerator.append(torch.logsumexp(marginal, 0))

                logp_trans_t = (alpha + beta) - logZs.unsqueeze(-1).unsqueeze(-1)
                logp_trans_t[indexes, lens-1] = float("-inf")
                logp_trans_t = torch.logsumexp(logp_trans_t, dim=1)
                trans_denominator.append(torch.logsumexp(logp_trans_t, 0))
            
            strans_numerator = [torch.logsumexp(torch.stack(strans_numerator), 0)]
            strans_denominator = [torch.logsumexp(torch.stack(strans_denominator,-1).squeeze(), -1)]

            etrans_numerator = [torch.logsumexp(torch.stack(etrans_numerator), 0)]
            etrans_denominator = [torch.logsumexp(torch.stack(etrans_denominator,-1).squeeze(), -1)]
            
            emits_numerator = [torch.logsumexp(torch.stack(emits_numerator, 0), 0)]
            emits_denominator = [torch.logsumexp(torch.stack(emits_denominator), 0)]

            if len(trans_denominator) > 1:
                trans_numerator = [torch.logsumexp(torch.stack(trans_numerator, 0), 0)]
                trans_denominator = [torch.logsumexp(torch.stack(trans_denominator), 0)]

        strans_numerator = strans_numerator[0]
        strans_denominator = strans_denominator[0]
        etrans_numerator = etrans_numerator[0]
        etrans_denominator = etrans_denominator[0]

        emits_numerator = emits_numerator[0]
        emits_denominator = emits_denominator[0]

        trans_numerator = trans_numerator[0]
        trans_denominator = trans_denominator[0]
        self.tagger.strans.data = strans_numerator - strans_denominator

        self.tagger.etrans.data = etrans_numerator- etrans_denominator
        self.tagger.emits.data = emits_numerator - emits_denominator.unsqueeze(-1)

        rows_to_keep_trans = ~torch.isfinite(trans_denominator)
        trans_denominator[rows_to_keep_trans] = 0
        new_trans = trans_numerator - trans_denominator.unsqueeze(-1)
        new_trans[rows_to_keep_trans] = self.tagger.trans.data[rows_to_keep_trans]
        self.tagger.trans.data = new_trans

        # to_keep_strans = ~torch.isfinite(strans_numerator)
        # self.tagger.strans.data = self.tagger.strans.data * to_keep_strans + torch.exp(strans_numerator - strans_denominator) + 1e-6
        # self.tagger.strans.data = self.tagger.strans.data / self.tagger.strans.data.sum()

        # to_keep_etrans = ~torch.isfinite(etrans_numerator)
        # self.tagger.etrans.data = self.tagger.etrans.data * to_keep_etrans + torch.exp(etrans_numerator- etrans_denominator) + 1e-6
        # self.tagger.etrans.data = self.tagger.etrans.data / self.tagger.etrans.data.sum()

        # column_to_keep_emits = ~torch.isfinite(emits_numerator) 
        # rows_to_keep_emits = ~torch.isfinite(emits_denominator)
        # emits_denominator[rows_to_keep_emits] = 0
        # to_keep_emits = column_to_keep_emits + rows_to_keep_emits.unsqueeze(-1)
        # self.tagger.emits.data = self.tagger.emits.data * to_keep_emits + torch.exp(emits_numerator - emits_denominator.unsqueeze(-1)) + 1e-6
        # self.tagger.emits.data = self.tagger.emits.data / self.tagger.emits.data.sum(-1).unsqueeze(-1)

        # rows_to_keep_trans = ~torch.isfinite(trans_denominator)
        # trans_denominator[rows_to_keep_trans] = 0
        # self.tagger.trans.data = self.tagger.trans.data * rows_to_keep_trans + torch.exp(trans_numerator - trans_denominator) + 1e-6
        # self.tagger.trans.data = self.tagger.trans.data / self.tagger.trans.data.sum(-1).unsqueeze(-1)


    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric, manyToOne = 0, AccuracyMethod(), ManyToOneAccuracy(self.vocab.n_labels)
        
        for words, chars, labels, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            margial = self.tagger.get_logZ(s_emit, mask)
            loss += -margial * words.size(0)

            predicts = self.tagger.viterbi(s_emit, mask)
            metric(predicts, targets)
            manyToOne(predicts, targets)

        loss /= len(loader.dataset)

        return float(loss), metric, manyToOne

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_labels = []
        for words, chars, possible_labels in loader:
            mask = words.ne(self.vocab.pad_index)
            lens = mask.sum(dim=1)
            targets = torch.split(labels[mask], lens.tolist())

            s_emit = self.tagger(words)
            s_emit[~possible_labels] = 0
            predicts = self.tagger.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

