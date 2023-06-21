# Type Hinting
from typing import List

# Pytest
import pytest
import torch

# user-defined libs.
from src.problems.solver import ODESolver, PDESolver


@pytest.mark.parametrize("use_odeint", [True, False])
def test_ode_solver(use_odeint: bool):
    zeta: float = 0.15
    omega0: float = 2.00

    # Create an instance of ODESolver
    # Problem: (Damped) Simple Harmonic Oscillator
    solver = ODESolver(
        equation="sho",
        params=(zeta, omega0),
        trange=(0, 10),
        Nt=1000,
    )

    # Define an example initial condition
    initial_condition: List = [1, 0]

    # Solve the ODE using solve_ivp
    solution = solver.solve_ivp(initial_condition, use_odeint=use_odeint)
    assert isinstance(solution, torch.Tensor)
    assert solution.shape == (len(initial_condition), solver.Nt)


@pytest.mark.parametrize("use_odeint", [True, False])
def test_pde_solver(use_odeint: bool):
    nu = 0.01 / torch.pi

    # Create an instance of ODESolver
    # Problem: (Damped) Simple Harmonic Oscillator
    solver = PDESolver(
        equation="burgers",
        params=(1, nu),
        xrange=(-1, 1),
        trange=(0, 1),
        Nx=256,
        Nt=128,
    )

    # Define an example initial condition
    initial_condition = lambda x: -torch.sin(torch.pi * x)

    # Solve the ODE using solve_ivp
    solution = solver.solve_ivp(initial_condition, use_odeint=use_odeint)
    assert isinstance(solution, torch.Tensor)
    assert solution.shape == (solver.Nx, solver.Nt)
