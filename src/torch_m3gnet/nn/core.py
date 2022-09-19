from __future__ import annotations

import torch


class GatedMLP(torch.nn.Module):
    """
    Note
    ----
    m3gnet.layers._core.GatedMLP
    """

    def __init__(
        self,
        in_features: int,
        dimensions: list[int],
        is_output: bool = False,
        use_bias: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.dimensions = dimensions
        self.is_output = is_output
        self.use_bias = use_bias

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
                    bias=self.use_bias,
                    device=device,
                )
            )
            if not (self.is_output and (i == num_layers - 1)):
                self.gate.append(torch.nn.SiLU())

            self.gate.append(
                torch.nn.Linear(
                    in_features=in_features_i,
                    out_features=out_features_i,
                    bias=self.use_bias,
                    device=device,
                )
            )
            if i == num_layers - 1:
                self.gate.append(torch.nn.Sigmoid())
            else:
                self.gate.append(torch.nn.SiLU())

    def forward(self, input):
        return self.dense(input) * self.gate(input)
