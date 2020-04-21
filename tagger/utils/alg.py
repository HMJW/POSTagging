# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


@torch.enable_grad()
def crf(scores, mask, target=None, partial=False):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    training = scores.requires_grad
    # always enable the gradient computation of scores
    # in order for the computation of marginal probs
    s_i, s_c = inside(scores.requires_grad_(), mask)
    logZ = s_c[0].gather(0, lens.unsqueeze(0)).sum()
    # marginal probs are used for decoding, and can be computed by
    # combining the inside algorithm and autograd mechanism
    # instead of the entire inside-outside process
    probs, = autograd.grad(logZ, scores, retain_graph=training)

    if target is None:
        return probs
    # the second inside process is needed if use partial annotation
    if partial:
        s_i, s_c = inside(scores, mask, target)
        score = s_c[0].gather(0, lens.unsqueeze(0)).sum()
    else:
        score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
    loss = logZ - score

    return loss, probs


def inside(scores, mask, cands=None):
    # the end position of each sentence in a batch
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    # [seq_len, seq_len, batch_size]
    scores = scores.permute(2, 1, 0)
    # include the first token of each sentence
    mask = mask.index_fill(1, lens.new_tensor(0), 1)
    # [seq_len, seq_len, batch_size]
    mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    s_c.diagonal().fill_(0)

    # set the scores of arcs excluded by cands to -inf
    if cands is not None:
        cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
        cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
        cands = cands.permute(2, 1, 0) & mask
        scores = scores.masked_fill(~cands, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        # diag_mask is used for ignoring the excess of each sentence
        # [batch_size, n]
        cand_mask = diag_mask = mask.diagonal(w)

        # ilr = C(i->r) + C(j->r+1)
        # [n, w, batch_size]
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        if cands is not None:
            cand_mask = torch.isfinite(ilr).any(1).t() & diag_mask
        ilr = ilr.permute(2, 0, 1)[cand_mask].logsumexp(-1)
        # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
        il = ilr + scores.diagonal(-w)[cand_mask]
        # fill the w-th diagonal of the lower triangular part of s_i
        # with I(j->i) of n spans
        s_i.diagonal(-w)[cand_mask] = il
        # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
        ir = ilr + scores.diagonal(w)[cand_mask]
        # fill the w-th diagonal of the upper triangular part of s_i
        # with I(i->j) of n spans
        s_i.diagonal(w)[cand_mask] = ir

        # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        if cands is not None:
            cand_mask = torch.isfinite(cl).any(1).t() & diag_mask
        cl = cl.permute(2, 0, 1)[cand_mask].logsumexp(-1)
        s_c.diagonal(-w)[cand_mask] = cl
        # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        if cands is not None:
            cand_mask = torch.isfinite(cr).any(1).t() & diag_mask
        cr = cr.permute(2, 0, 1)[cand_mask].logsumexp(-1)
        s_c.diagonal(w)[cand_mask] = cr
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')

    return s_i, s_c


def eisner(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(mask.device))
    mask[:,0] = 1
    predicts = torch.cat(predicts)
    result = predicts.new_zeros(batch_size, seq_len)
    result = result.masked_scatter_(mask, predicts)
    mask[:,0] = 0
    return result

def backtrack(p_i, p_c, heads, i, j, complete):
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.

    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)
