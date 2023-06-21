import torch
import torch.nn as nn


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int) -> None:
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = 1 / (in_channels * out_channels)
        # (in_channel, out_channel, mode)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input: torch.tensor, weights: torch.tensor) -> torch.tensor:
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (batch, width, resolution)
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        # x_ft: (batch, width, modes)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        # out_ft: (batch, out_channel, 0.5*resolution + 1)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], self.weights1)

        # Return to physical space
        # x: (batch, width, resolution)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1D_block(nn.Module):
    def __init__(self, width: int, modes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.conv = nn.Conv1d(width, width, 1)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        x = x1 + x2
        return x


class Fourier_Feature(nn.Module):
    # Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
    # https://arxiv.org/abs/2006.10739
    def __init__(self, input_size: int = 2, mapping_size: int = 64, scale: float = 10, *args, **kwargs) -> None:
        super().__init__()
        self.B = nn.Parameter(scale * torch.randn(mapping_size//2, input_size), requires_grad=False)

    def forward(self, x):
        x_proj = (2 * torch.pi * x) @ self.B.to(x).T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)