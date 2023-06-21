# Type Hinting
from typing import Tuple, Callable, Union
from numpy.typing import ArrayLike

# Computing
from scipy.integrate import odeint, solve_ivp

# PyTorch
import torch

# Misc.
from dataclasses import dataclass

# user-defined libs.
from src.problems import ode
from src.problems import pde


@dataclass
class ODESolver:
    """
    ODESolver class provides a simple interface for solving ordinary differential equations (ODEs).

    Args:
        equation (Union[str, Callable]): The name of the equation representing the ODE or a callable representing the ODE.
        params (Tuple): Parameters to be passed to the ODE function.
        xrange (Tuple[float, float]): Range of x values.
        trange (Tuple[float, float]): Range of t values.
        Nt (int): Number of t values.

    Attributes:
        equation (Union[str, Callable]): The name of the equation representing the ODE or a callable representing the ODE.
        params (Tuple): Parameters to be passed to the ODE function.
        trange (Tuple[float, float]): Range of t values.
        Nt (int): Number of t values.
        ts (torch.Tensor): Tensor containing t values.

    Methods:
        solve_ivp: Solves the ODE using the selected solver.

    """

    equation: Union[str, Callable]
    params: Tuple
    trange: Tuple[float, float] = (0, 1)
    Nt: int = 0

    @property
    def ts(self) -> torch.Tensor:
        """
        Generate a tensor of t values.

        Returns:
            torch.Tensor: Tensor containing t values.
        """
        return torch.linspace(*self.trange, self.Nt)

    def solve_ivp(
        self,
        initial_condition: ArrayLike,
        use_odeint: bool = False,
    ) -> ArrayLike:
        """
        Solve the ODE using the selected solver.

        Args:
            initial_condition (Union[ArrayLike, Callable]): The initial condition of the ODE.
                If it is a callable, it is evaluated using xs to obtain the initial condition values.
                If it is an ArrayLike, it is used as the initial condition directly.
            use_odeint (bool, optional): Whether to use odeint for solving the ODE.
                If False, solve_ivp method is used. Defaults to False.

        Returns:
            ArrayLike: The solution of the ODE.

        Raises:
            TypeError: If the initial condition is not callable or an ArrayLike object.

        """
        if isinstance(self.equation, str):
            func = getattr(ode, self.equation)
        else:
            func = self.equation

        if use_odeint:
            solution = odeint(func, initial_condition, self.ts, args=self.params, tfirst=True).T
        else:
            solution = solve_ivp(
                func, self.trange, initial_condition, args=self.params, method="LSODA", t_eval=self.ts
            ).y

        return torch.Tensor(solution)


@dataclass
class PDESolver:
    """
    ODESolver class provides a simple interface for solving ordinary differential equations (ODEs).

    Args:
        equation (Union[str, Callable]): The name of the equation representing the ODE or a callable representing the ODE.
        params (Tuple): Parameters to be passed to the ODE function.
        xrange (Tuple[float, float]): Range of x values.
        trange (Tuple[float, float]): Range of t values.
        Nt (int): Number of t values.

    Attributes:
        equation (Union[str, Callable]): The name of the equation representing the ODE or a callable representing the ODE.
        params (Tuple): Parameters to be passed to the ODE function.
        trange (Tuple[float, float]): Range of t values.
        Nt (int): Number of t values.
        ts (torch.Tensor): Tensor containing t values.

    Methods:
        solve_ivp: Solves the ODE using the selected solver.

    """

    equation: Union[str, Callable]
    params: Tuple
    xrange: Tuple[float, float] = (0, 1)
    trange: Tuple[float, float] = (0, 1)
    Nx: int = 0
    Nt: int = 0

    @property
    def xs(self) -> torch.Tensor:
        """
        Generate a tensor of x values.

        Returns:
            torch.Tensor: Tensor containing x values.
        """
        return torch.linspace(*self.xrange, self.Nx)

    @property
    def ts(self) -> torch.Tensor:
        """
        Generate a tensor of t values.

        Returns:
            torch.Tensor: Tensor containing t values.
        """
        return torch.linspace(*self.trange, self.Nt)

    def solve_ivp(
        self,
        initial_condition: Union[ArrayLike, Callable],
        use_odeint: bool = False,
    ) -> ArrayLike:
        """
        Solve the ODE using the selected solver.

        Args:
            initial_condition (Union[ArrayLike, Callable]): The initial condition of the ODE.
                If it is a callable, it is evaluated using xs to obtain the initial condition values.
                If it is an ArrayLike, it is used as the initial condition directly.
            use_odeint (bool, optional): Whether to use odeint for solving the ODE.
                If False, solve_ivp method is used. Defaults to False.

        Returns:
            ArrayLike: The solution of the ODE.

        Raises:
            TypeError: If the initial condition is not callable or an ArrayLike object.

        """
        if isinstance(initial_condition, Callable):
            initial_condition = initial_condition(self.xs)
        if isinstance(self.equation, str):
            func = getattr(pde, self.equation)
        else:
            func = self.equation

        if use_odeint:
            solution = odeint(func, initial_condition, self.ts, args=self.params, tfirst=True).T
        else:
            solution = solve_ivp( func, self.trange, initial_condition, args=self.params, method="LSODA", t_eval=self.ts).y

        return torch.Tensor(solution)
