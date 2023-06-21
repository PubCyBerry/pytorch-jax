from typing import Any, List
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from src.models.model_utils import get_activation
from src.models.model_layers import FNO1D_block, Fourier_Feature


class BaseNN(nn.Module):
    def __init__(self, init_type: str = "xavier_truncated", *args, **kwds) -> None:
        super().__init__()
        self.apply(self._init_weights(init_type))

    def forward(self, *args: Any, **kwds: Any):
        raise NotImplementedError

    def _init_weights(self, init_type: str = "xavier_truncated"):
        """Define how to initialize weights and biases for each type of layer."""

        def xavier_truncated(module):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            std = np.sqrt(2.0 / (fan_in + fan_out))
            nn.init.trunc_normal_(module.weight.data, std=std, mean=0, a=-2, b=2)

        def he_truncated(module):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            std = np.sqrt(2.0 / fan_in)
            nn.init.trunc_normal_(module.weight.data, std=std, mean=0, a=-2, b=2)

        init_dict = {"xavier_truncated": xavier_truncated, "he_truncated": he_truncated}

        def _initializer(module):
            if isinstance(module, nn.Linear):
                init_func = init_dict.get(init_type)
                if init_func is not None:
                    init_func(module)
                if module.bias is not None:
                    module.bias.data.zero_()

        return _initializer


class MLP(BaseNN):
    def __init__(
        self,
        layers: List[int] = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        activation: str = "tanh",
        linear_output: bool = False,
        init_type: str = "xavier_truncated",
    ) -> None:
        super().__init__()
        modules = list()

        for i, (in_f, out_f) in enumerate(zip(layers, layers[1:])):
            modules.append(("linear %d" % i, nn.Linear(in_f, out_f)))
            modules.append(("%s %d" % (activation, i), get_activation(activation)))
        if linear_output:
            modules.pop()
        self.net = nn.Sequential(OrderedDict(modules))
        self.apply(self._init_weights(init_type))

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.net(x)


class MLP_concat(MLP):
    def __init__(
        self,
        layers: List[int] = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        activation: str = "tanh",
        linear_output: bool = False,
        init_type: str = "xavier_truncated",
    ) -> None:
        super().__init__(layers, activation, linear_output, init_type)

    def forward(self, x, t, *args: Any, **kwargs: Any) -> Any:
        return self.net(torch.cat([x, t], dim=1))


class DeepONet(BaseNN):
    def __init__(
        self,
        branch_layers: List[int] = [128, 20, 20, 20, 20],
        trunk_layers: List[int] = [2, 20, 20, 20, 20],
        activation: str = "relu",
        init_type: str = "xavier_truncated",
    ) -> None:
        super().__init__()
        self.num_sensor: int = branch_layers[0]

        self.branch_net = MLP(branch_layers, activation, linear_output=True)
        self.trunk_net = MLP(trunk_layers, activation, linear_output=True)
        self.b0 = nn.Parameter(torch.zeros((1,)), requires_grad=True)

        self.apply(self._init_weights(init_type))

    def forward(self, u, y=None) -> torch.Tensor:
        """
        u: (batch, num_input_sensor)
        y: (batch, num_output_sensor, dim_output_sensor)
        """
        if y is None:
            u, y = u
        if y.ndim == 2:  # no batch dim
            u = u.unsqueeze(0)
            y = y.unsqueeze(0)
        # (batch, num_input_sensors) -> (batch, hidden_dim)
        b = self.branch_net(u)
        # (batch, num_output_sensors, dim_output_sensors) -> (batch, num_output_sensors, hidden_dim)
        t = self.trunk_net(y)
        # (batch, num_output_sensors)
        s = torch.einsum("bi, bni -> bn", b, t) + self.b0
        return s


class FNO1D(BaseNN):
    def __init__(
        self,
        num_step: int = 1,
        n_dimension: int = 1,
        modes: int = 16,
        width: int = 32,
        num_blocks: int = 3,
        hidden_dim: int = 128,
        activation: str = "relu",
        init_type: str = "xavier_truncated",
    ) -> None:
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.num_step = num_step
        self.n_dimension = n_dimension

        self.modes1: int = modes
        self.width: int = width
        self.fc0 = nn.Linear(num_step + n_dimension, self.width)  # input channel is : (a(x,t[ti ~ to]), x)

        modules = list()
        for i in range(num_blocks):
            modules.append(("fno1d_block %d" % i, FNO1D_block(width, modes)))
            modules.append(("%s %d" % (activation, i), get_activation(activation)))
        modules.pop()

        self.fno_layers = nn.Sequential(OrderedDict(modules))

        self.fc1 = nn.Linear(self.width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.activation = get_activation(activation)

        self.apply(self._init_weights(init_type))

    def forward(self, a, x=None, total_step=None):
        if x is None or total_step is None:
            a, x, total_step = a

        if x.ndim != 3:
            a = a.unsqueeze(0)
            x = x.unsqueeze(0)

        if not isinstance(total_step, int):
            total_step = int(total_step[0])

        # step = 1
        preds = a  # t = t_init (batch, grid_x, num_step)
        for t in range(total_step - self.num_step):
            # (batch, grid_x, 1)
            im = self.forward_step(a, x)
            # (batch, grid_x, num_step + t)
            preds = torch.cat([preds, im], -1)
            # (batch, grid_x, num_step), (batch, grid_x, n_dimension)
            a = preds[..., t + 1 :]

        return preds  # (batch, grid_x, total_step)

    def forward_step(self, a, x) -> torch.tensor:
        # (batch, resolution, num_step + n_dimension) -> (batch, resolution, width)
        x = self.fc0(torch.cat([a, x], -1))
        # (batch, width, resolution)
        x = x.permute(0, 2, 1)

        x = self.fno_layers(x)

        # (batch, resolution, width)
        x = x.permute(0, 2, 1)
        # (batch, resolution, hidden_dim)
        x = self.fc1(x)
        x = self.activation(x)
        # (batch, resolution, 1)
        x = self.fc2(x)
        return x
