from __future__ import annotations

import torch


class GatedMLP(torch.nn.Module):
    def __init__(self, in_features: int, num_layers: int):
        super().__init__()
        self.in_features = in_features
        self.num_layers = num_layers

        self.dense = torch.nn.Sequential()
        for _ in range(num_layers):
            self.dense.append(torch.nn.Linear(in_features=in_features, out_features=in_features))
            self.dense.append(torch.nn.Sigmoid())

        self.gate = torch.nn.Sequential()
        for i in range(num_layers):
            self.gate.append(torch.nn.Linear(in_features=in_features, out_features=in_features))
            if i == num_layers - 1:
                self.gate.append(torch.nn.SiLU())
            else:
                self.gate.append(torch.nn.Sigmoid())

    def forward(self, input):
        return self.dense(input) * self.gate(input)
