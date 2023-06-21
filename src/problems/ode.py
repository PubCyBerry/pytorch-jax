from typing import List

from numpy.typing import ArrayLike


def sho(t: float, X: ArrayLike, zeta: float, omega0: float) -> List[float]:
    """Solves the Ordinary Differential Equation (ODE) for a free harmonic oscillator.

    Args:
        t (float): The independent variable (time).
        X (List[float]): A list containing the variables x and dx.
        zeta (float): The damping ratio of the oscillator.
        omega0 (float): The natural frequency of the oscillator.

    Returns:
        List[float]: A list containing the derivatives of x and dx.

    Comments:
        This function represents the ODE for a free harmonic oscillator.
        The ODE describes the motion of a mass-spring system with damping.
        The equation is of the form: d^2x/dt^2 + 2*zeta*omega0*dx/dt + omega0^2*x = 0.
        The function returns the derivatives of x and dx, which can be integrated
        numerically to obtain the solution of the ODE.
    """
    x, dx = X
    f = [dx, -2 * zeta * omega0 * dx - omega0**2 * x]
    return f
