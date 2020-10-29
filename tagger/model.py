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
        strans_numerator = torch.full([self.tagger.n_tags], float("-inf")).to(device)
        etrans_numerator = torch.full([self.tagger.n_tags], float("-inf")).to(device)
        emits_numerator = torch.full([self.tagger.n_tags, self.tagger.n_words], float("-inf")).to(device)
        trans_numerator = torch.full([self.tagger.n_tags, self.tagger.n_tags], float("-inf")).to(device)

        for words, labels in loader:
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
            posteriors = gamma - logZs.unsqueeze(-1).unsqueeze(-1)
            posteriors[~mask] = float('-inf')

            strans_posterior = torch.logsumexp(posteriors[indexes, 0], dim=0)
            strans_numerator = torch.logaddexp(strans_numerator, strans_posterior)

            etrans_posterior = torch.logsumexp(posteriors[indexes, lens-1], dim=0)
            etrans_numerator = torch.logaddexp(etrans_numerator, etrans_posterior)

            count = torch.arange(self.tagger.n_words).unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1).to(words.device) == words.unsqueeze(-1)
            count[~mask] = 0
            # exp may loss accuracy, use einsum to save memory
            count_posteriors = torch.log(torch.einsum("bxy, byz->bxz", torch.exp(posteriors.transpose(1,2)), count.float()))
            emits_numerator = torch.logaddexp(emits_numerator, torch.logsumexp(count_posteriors, 0))

            # B*T-1*N*1 + B*T-1*1*N + 1*1*N*N + B*T-1*1*N
            if (lens > 1).any():
                log_xi_sum = alpha[:, 0: max_len-1].unsqueeze(-1) + beta[:, 1:max_len].unsqueeze(2) + self.tagger.trans.unsqueeze(0).unsqueeze(0) + s_emit[:,1:max_len].unsqueeze(2)
                log_xi_sum = torch.logsumexp(log_xi_sum, dim=1) - logZs.unsqueeze(-1).unsqueeze(-1)
                torch.logaddexp(trans_numerator, torch.logsumexp(log_xi_sum, 0))
    
        strans_numerator_norm = strans_numerator - torch.logsumexp(strans_numerator, dim=-1)
        etrans_numerator_norm = etrans_numerator - torch.logsumexp(etrans_numerator, dim=-1)
        self.tagger.strans.data = strans_numerator_norm
        self.tagger.etrans.data = etrans_numerator_norm

        emits_numerator_norm = emits_numerator - torch.logsumexp(emits_numerator,dim=-1).unsqueeze(-1)
        self.tagger.emits.data = emits_numerator_norm

        trans_numerator_sum = torch.logsumexp(trans_numerator, dim=-1)
        rows_to_keep_trans = ~torch.isfinite(trans_numerator_sum)
        trans_numerator_sum[rows_to_keep_trans] = 0
        new_trans = trans_numerator - trans_numerator_sum.unsqueeze(-1)
        new_trans[rows_to_keep_trans] = self.tagger.trans.data[rows_to_keep_trans]
        self.tagger.trans.data = new_trans


    @torch.no_grad()
    def evaluate(self, loader):
        self.tagger.eval()

        loss, metric, manyToOne = 0, AccuracyMethod(), ManyToOneAccuracy(self.vocab.n_labels)
        
        for words, labels in loader:
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
        for words in loader:
            words = next(words)
            mask = words.ne(self.vocab.pad_index)
            s_emit = self.tagger(words)
            predicts = self.tagger.viterbi(s_emit, mask)
            all_labels.extend(predicts)

        all_labels = [self.vocab.id2label(seq.tolist()) for seq in all_labels]
        return all_labels

