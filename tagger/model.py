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
        strans_grad = [torch.full([self.tagger.n_tags], 0)]
        etrans_grad = [torch.full([self.tagger.n_tags], 0)]
        weight_grad = [torch.full([self.tagger.n_features, self.tagger.n_tags], 0)]
        trigram_grad = [torch.full([self.config.n_trigrams], 0)]
        trans_grad = [torch.full([self.tagger.n_tags, self.tagger.n_tags], 0)]

        for words, labels in loader:
            mask = words.ge(0).unsqueeze(0)
            length = len(words)
            s_emit = self.tagger(words, self.vocab)
            s_emit = s_emit.unsqueeze(0)

            # B*T*N
            alpha = self.tagger.forw(s_emit, mask)[0]
            beta = self.tagger.back(s_emit, mask)[0]
            logZ = torch.logsumexp(alpha[length-1,:] + self.tagger.etrans, dim=0)
            assert torch.isfinite(logZ).all()
            gamma = alpha + beta

            
            posteriors = gamma - logZs.unsqueeze(-1).unsqueeze(-1)
            posteriors[~mask] = float('-inf')

            strans_posterior = torch.logsumexp(posteriors[indexes, 0], dim=0)
            strans_numerator.append(strans_posterior)

            etrans_posterior = torch.logsumexp(posteriors[indexes, lens-1], dim=0)
            etrans_numerator.append(etrans_posterior)


            feature_counts = self.tagger.all_words_features.unsqueeze(-1) == torch.arange(self.tagger.n_features).unsqueeze(0).unsqueeze(0).to(posteriors.device)
            feature_counts[self.tagger.all_words_features == 0] = 0
            all_feature_counts = feature_counts.sum(1).unsqueeze(-1).expand(-1, -1, self.tagger.n_tags)
            feature_counts = all_feature_counts.unsqueeze(0).expand(batch_size, -1, -1, -1)
            feature_counts = feature_counts.gather(1, words.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.tagger.n_features, self.tagger.n_tags))
            feature_counts[~mask] = 0
            partial_Z = (torch.exp(emits.transpose(0, 1)).unsqueeze(1)* all_feature_counts).sum(0)
            feature_partial = (feature_counts - partial_Z.unsqueeze(0).unsqueeze(0)) * torch.exp(posteriors).unsqueeze(2)
            gradients_w = feature_partial.sum(0)
            gradients_w = gradients_w.sum(0)
            weight_grad.append(gradients_w)

            trigram_counts = self.tagger.all_words_trigrams.unsqueeze(-1) == torch.arange(self.tagger.n_trigrams).unsqueeze(0).unsqueeze(0).to(posteriors.device)
            trigram_counts[self.tagger.all_words_trigrams == 0] = 0
            all_trigram_counts = trigram_counts.sum(1)

            trigram_counts = all_trigram_counts.unsqueeze(0).expand(batch_size, -1, -1)
            trigram_counts = trigram_counts.gather(1, words.unsqueeze(-1).expand(-1, -1, self.tagger.n_trigrams))
            trigram_counts[~mask] = 0
            partial_Z = (torch.exp(emits.transpose(0, 1)).unsqueeze(-1)* all_trigram_counts.unsqueeze(1)).sum(0).sum(0)
            trigram_partial = (trigram_counts - partial_Z.unsqueeze(0).unsqueeze(0)) * torch.exp(posteriors.sum(-1)).unsqueeze(2)
            gradients_f = trigram_partial.sum(0)
            gradients_f = gradients_f.sum(0)
            trigram_grad.append(gradients_f)

            # B*N*1 + B*1*N + B*N*N + B*1*N
            if (lens > 1).any():
                log_xi_sum = [alpha[:, l].unsqueeze(-1) + beta[:, l+1].unsqueeze(1) + self.tagger.trans.unsqueeze(0) + s_emit[:,l+1].unsqueeze(1) for l in range(max_len-1)]
                log_xi_sum = torch.stack(log_xi_sum, 0)
                log_xi_sum = torch.logsumexp(log_xi_sum, dim=0) - logZs.unsqueeze(-1).unsqueeze(-1)
                trans_numerator.append(torch.logsumexp(log_xi_sum, 0))

            strans_numerator = [torch.logsumexp(torch.stack(strans_numerator), 0)]
            etrans_numerator = [torch.logsumexp(torch.stack(etrans_numerator), 0)]
            weight_grad = [torch.sum(torch.stack(weight_grad, 0), 0)]
            trigram_grad = [torch.sum(torch.stack(trigram_grad, 0), 0)]

            if len(trans_numerator) > 1:
                trans_numerator = [torch.logsumexp(torch.stack(trans_numerator, 0), 0)]

        strans_numerator = strans_numerator[0]
        strans_numerator_norm = strans_numerator - torch.logsumexp(strans_numerator, dim=-1)
        etrans_numerator = etrans_numerator[0]
        etrans_numerator_norm = etrans_numerator - torch.logsumexp(etrans_numerator, dim=-1)
        self.tagger.strans.data = strans_numerator_norm
        self.tagger.etrans.data = etrans_numerator_norm

        weight_grad = weight_grad[0]
        self.tagger.weights.data += weight_grad

        trigram_grad = trigram_grad[0]
        self.tagger.trigram_weights.data += trigram_grad

        trans_numerator = trans_numerator[0]
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
        
        emits = self.tagger.get_emits(self.vocab)
        for words, labels in loader:
            length = len(words)

            s_emit = self.tagger(words, emits)
            s_emit, mask = s_emit.unsqueeze(0), words.ge(0).unsqueeze(0)
            margial = self.tagger.get_logZ(s_emit, mask)
            loss += -margial

            predicts = self.tagger.viterbi(s_emit, mask)
            labels = labels.unsqueeze(0)
            metric(predicts, labels)
            manyToOne(predicts, labels)

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

