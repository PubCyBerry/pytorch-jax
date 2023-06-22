import numpy as np
from numpy.typing import ArrayLike


def fft_diff(x: ArrayLike, order: int = 1, period: float = 2 * np.pi):
    """Spectral Differentiation.

    Args:
        x (jax.Array): 1D sequence to be differentiated.
        order (int, optional): The order of differentiation. Defaults to 1.
        period (float, optional): The assumed period of the sequence. Defaults to 2*pi.
    """
    Nx: int = len(x)
    dx: float = period / Nx
    # wave number
    k = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    # Spatial Derivative in the Fourier domain
    x_hat = np.power(1j * k, order) * np.fft.fft(x)
    # Switching in the spatial domain
    x_diff = np.fft.ifft(x_hat)
    return x_diff.real


# Define pseudo-spectral solver
# PDE -> FFT -> ODE
def burgers(t: float, u: ArrayLike, period: float, nu: float):
    """solve Burgers equation with spectral method.

    u_t + u * u_x = \nu u_xx
    u: u(x, t_i), shape: (num_x)
    period: assumed period of sequence
    coefficient, nu: kinematic viscosity
    """
    u_x = fft_diff(u, order=1, period=period)  # first derivative
    u_xx = fft_diff(u, order=2, period=period)  # second derivative
    u_t = -u * u_x + nu * u_xx

    return u_t


# Define pseudo-spectral solver
# PDE -> FFT -> ODE
def kdv(t: float, u: ArrayLike, period: float, coefficient: float):
    r"""Solve Korteweg de Vries equation with spectral method.

    u_t + u * u_x + \delta^2 * u_xxx = 0
    u: u(x, t_i), shape: (num_x)
    period: assumed period of sequence
    coefficient: kinematic viscosity
    """
    u_x = fft_diff(u, order=1, period=period)
    u_xxx = fft_diff(u, order=3, period=period)
    u_t = -u * u_x - coefficient * u_xxx
    return u_t
