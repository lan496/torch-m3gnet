from __future__ import annotations

import torch


class GatedMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        dimensions: list[int],
        is_output: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.dimensions = dimensions
        self.is_output = is_output

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
                torch.nn.Linear(
                    in_features=in_features_i,
                    out_features=out_features_i,
                    device=device,
                )
            )
            if (i < num_layers - 1) and (not self.is_output):
                self.dense.append(torch.nn.Sigmoid())

            self.gate.append(
                torch.nn.Linear(
                    in_features=in_features_i,
                    out_features=out_features_i,
                    device=device,
                )
            )
            if i == num_layers - 1:
                self.gate.append(torch.nn.SiLU())
            else:
                self.gate.append(torch.nn.Sigmoid())

    def forward(self, input):
        return self.dense(input) * self.gate(input)
