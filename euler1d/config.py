"""Configuration dataclasses for the Euler 1D solver.

Immutable data containers defining the physical and numerical parameters of a
simulation: gas thermodynamic properties, computational mesh, time-stepping
control, initial conditions (Riemann problems or smooth test cases), and
boundary condition type.

See Also
--------
docs/01_euler_equations.md : Chapter 1 — physical meaning of the gas
    properties, primitive variables, and Riemann problem structure.

References
----------
[Toro, 2009] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid
    Dynamics, 3rd ed., Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class GasProperties:
    """Thermodynamic properties of a perfect gas.

    Attributes
    ----------
    gamma : float
        Ratio of specific heats (cp/cv). Controls the compressibility of the
        gas and the sound speed. Default 1.4 corresponds to diatomic gases
        (air) at standard conditions. See docs/01_euler_equations.md section
        1.2.
    """

    gamma: float = 1.4

    @property
    def gm1(self) -> float:
        return self.gamma - 1.0

    @property
    def gp1(self) -> float:
        return self.gamma + 1.0


@dataclass(frozen=True)
class MeshConfig:
    """Uniform 1D computational mesh.

    Attributes
    ----------
    x_min : float
        Left boundary of the computational domain.
    x_max : float
        Right boundary of the computational domain.
    n_cells : int
        Number of finite volume cells. Cell size dx = (x_max - x_min) / n_cells.
    """

    x_min: float
    x_max: float
    n_cells: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.n_cells


@dataclass(frozen=True)
class TimeConfig:
    """Time-stepping parameters.

    Attributes
    ----------
    t_final : float
        Physical end time of the simulation.
    cfl : float
        Courant-Friedrichs-Lewy number controlling the adaptive time step
        size. The time step is dt = cfl * dx / max_wave_speed.
    dt_fixed : float or None
        If set, overrides the adaptive CFL-based time step with a fixed value.
    """

    t_final: float
    cfl: float = 0.9
    dt_fixed: float | None = None


@dataclass(frozen=True)
class PrimitiveState:
    """Uniform thermodynamic state in primitive variables.

    Represents a constant flow state used to define initial conditions
    (left/right states of a Riemann problem). See docs/01_euler_equations.md
    section 1.3.

    Attributes
    ----------
    rho : float
        Density — mass per unit volume.
    u : float
        Velocity — bulk fluid velocity.
    p : float
        Pressure — thermodynamic pressure.
    """

    rho: float
    u: float
    p: float


@dataclass(frozen=True)
class RiemannProblem:
    """Initial condition defined by two constant states separated by a discontinuity.

    The Riemann problem is the fundamental building block of Godunov-type finite
    volume methods. Its solution consists of three waves (two acoustic, one
    entropy) connecting four constant states. See docs/01_euler_equations.md
    section 1.5.

    Attributes
    ----------
    left : PrimitiveState
        Uniform state to the left of the initial discontinuity.
    right : PrimitiveState
        Uniform state to the right of the initial discontinuity.
    x_discontinuity : float
        Position of the initial discontinuity within the domain.
    """

    left: PrimitiveState
    right: PrimitiveState
    x_discontinuity: float = 0.5


@dataclass(frozen=True)
class SmoothProblem:
    """Initial condition defined by smooth analytical functions.

    Used for convergence studies where an exact solution is known at all times
    (e.g., advected entropy wave, isentropic acoustic wave). Periodic boundary
    conditions are typically required.

    Attributes
    ----------
    init_fn : callable
        Function (x, gamma) -> (rho, u, p) returning the initial primitive
        state arrays.
    exact_fn : callable
        Function (x, t, gamma) -> (rho, u, p) returning the exact primitive
        state arrays at any time t, for computing error norms.
    label : str
        Short descriptive label for display purposes.
    """

    init_fn: Callable  # (x, gamma) -> (rho, u, p) arrays
    exact_fn: Callable  # (x, t, gamma) -> (rho, u, p) arrays
    label: str = "smooth"


@dataclass
class SimulationConfig:
    """Complete specification of a simulation.

    Aggregates all parameters needed to run a simulation: gas properties,
    mesh, time-stepping, initial conditions, and boundary treatment. This is
    the only mutable configuration class, allowing incremental construction.

    Attributes
    ----------
    gas : GasProperties
        Thermodynamic properties of the working gas.
    mesh : MeshConfig
        Computational domain and spatial discretization.
    time : TimeConfig
        End time and CFL-based time step control.
    problem : RiemannProblem or SmoothProblem
        Initial condition specification.
    bc_type : str
        Boundary condition type: 'transmissive', 'reflective', or 'periodic'.
    """

    gas: GasProperties = field(default_factory=GasProperties)
    mesh: MeshConfig = field(default_factory=lambda: MeshConfig(0.0, 1.0, 100))
    time: TimeConfig = field(default_factory=lambda: TimeConfig(0.2))
    problem: RiemannProblem | SmoothProblem = field(
        default_factory=lambda: RiemannProblem(
            PrimitiveState(1.0, 0.0, 1.0),
            PrimitiveState(0.125, 0.0, 0.1),
        )
    )
    bc_type: str = "transmissive"
