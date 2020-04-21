# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, n_in):
        super(Attention, self).__init__()

        self.n_in = n_in
        self.weight = nn.Parameter(torch.Tensor(n_in, 1))

        self.reset_parameters()

    def extra_repr(self):
        return f"n_in={self.n_in}"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x):
        p = x @ self.weight
        x = x.transpose(-1, -2) @ p.softmax(dim=-2)
        x = x.squeeze(-1)

        return x
