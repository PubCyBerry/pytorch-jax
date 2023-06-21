import torch.nn as nn


def get_activation(name: str = "lrelu") -> nn.Module:
    # choose activation function
    activations = nn.ModuleDict(
        [
            ["lrelu", nn.LeakyReLU(0.1)],
            ["relu", nn.ReLU()],
            ["tanh", nn.Tanh()],
            ["sigmoid", nn.Sigmoid()],
            ["gelu", nn.GELU()],
        ]
    )
    return activations[name]
