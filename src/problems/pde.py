from numpy.typing import ArrayLike
from scipy import fftpack


# Define pseudo-spectral solver
# PDE -> FFT -> ODE
def burgers(t: float, u: ArrayLike, period: float, nu: float):
    """solve Burgers equation with spectral method.

    u_t + u * u_x = \nu u_xx
    u: u(x, t_i), shape: (num_x)
    period: assumed period of sequence
    coefficient, nu: kinematic viscosity
    """
    u_x = fftpack.diff(u, order=1, period=period)  # first derivative
    u_xx = fftpack.diff(u, order=2, period=period)  # second derivative
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
    u_x = fftpack.diff(u, order=1, period=period)
    u_xxx = fftpack.diff(u, order=3, period=period)
    u_t = -u * u_x - coefficient * u_xxx
    return u_t
