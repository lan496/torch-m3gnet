from __future__ import annotations

import torch


class GatedMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        dimensions: list[int],
    ):
        super().__init__()
        self.in_features = in_features
        self.dimensions = dimensions

        self.dense = torch.nn.Sequential()
        self.gate = torch.nn.Sequential()
        num_layers = len(dimensions)
        for i in range(num_layers):
            if i == 0:
                in_features_i = self.in_features
            else:
                in_features_i = dimensions[i - 1]
            out_features_i = dimensions[i]

            self.dense.append(
                torch.nn.Linear(in_features=in_features_i, out_features=out_features_i)
            )
            self.dense.append(torch.nn.Sigmoid())

            self.gate.append(
                torch.nn.Linear(in_features=in_features_i, out_features=out_features_i)
            )
            if i == num_layers - 1:
                self.gate.append(torch.nn.SiLU())
            else:
                self.gate.append(torch.nn.Sigmoid())

    def forward(self, input):
        return self.dense(input) * self.gate(input)
