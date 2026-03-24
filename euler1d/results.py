"""Result processing: DataFrames, exact solution, error norms, convergence studies.

This module provides the post-processing pipeline for simulation results:

* Converting raw conservative-variable snapshots to human-readable DataFrames.
* Computing exact reference solutions (Riemann solver or analytic).
* Measuring discretisation errors with L1, L2, and Linf norms.
* Running mesh-convergence studies and estimating convergence orders.

See docs/08_test_cases.md, Section 8.6 for the mathematical background.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from euler1d.config import RiemannProblem, SimulationConfig, SmoothProblem
from euler1d.physics import conservative_to_primitive, sound_speed
from euler1d.riemann import sample_riemann
from euler1d.solver import SolverResult, run_simulation

if TYPE_CHECKING:
    from euler1d.schemes.base import NumericalScheme


def result_to_dataframe(result: SolverResult, time_index: int = -1) -> pd.DataFrame:
    """Convert a simulation snapshot to a pandas DataFrame.

    Extracts the conservative variables from the requested time snapshot,
    converts them to primitive variables (rho, u, p), and computes derived
    quantities (total energy E, Mach number).

    Parameters
    ----------
    result : SolverResult
        Output of :func:`euler1d.solver.run_simulation`.
    time_index : int, optional
        Index into ``result.snapshots`` (default -1, i.e. the final snapshot).

    Returns
    -------
    pd.DataFrame
        Columns: x, rho, u, p, E, mach.
    """
    U = result.snapshots[time_index]
    gamma = result.config.gas.gamma
    rho, u, p = conservative_to_primitive(U, gamma)
    a = sound_speed(rho, p, gamma)
    E = U[2]
    mach = np.abs(u) / a

    return pd.DataFrame({
        "x": result.x,
        "rho": rho,
        "u": u,
        "p": p,
        "E": E,
        "mach": mach,
    })


def compute_exact_solution(config: SimulationConfig) -> pd.DataFrame:
    """Compute the exact solution at t_final on the mesh.

    For Riemann problems, delegates to the exact Riemann solver
    (:func:`euler1d.riemann.sample_riemann`).  For smooth problems,
    evaluates the analytic ``exact_fn(x, t, gamma)`` provided by the
    test case (see docs/08_test_cases.md, Sections 8.4-8.5).

    Parameters
    ----------
    config : SimulationConfig
        Must contain either a ``RiemannProblem`` or a ``SmoothProblem``.

    Returns
    -------
    pd.DataFrame
        Columns: x, rho, u, p, E, mach -- evaluated at cell centres.
    """
    mesh = config.mesh
    gas = config.gas
    prob = config.problem
    dx = mesh.dx
    x = np.linspace(mesh.x_min + 0.5 * dx, mesh.x_max - 0.5 * dx, mesh.n_cells)

    if isinstance(prob, SmoothProblem):
        rho, u, p = prob.exact_fn(x, config.time.t_final, gas.gamma)
    else:
        rho, u, p = sample_riemann(
            x, config.time.t_final,
            prob.left.rho, prob.left.u, prob.left.p,
            prob.right.rho, prob.right.u, prob.right.p,
            gas.gamma,
            x0=prob.x_discontinuity,
        )
    E = p / gas.gm1 + 0.5 * rho * u**2
    a = sound_speed(rho, p, gas.gamma)

    return pd.DataFrame({
        "x": x,
        "rho": rho,
        "u": u,
        "p": p,
        "E": E,
        "mach": np.abs(u) / a,
    })


def compute_errors(
    result: SolverResult, exact: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Compute L1, L2, and Linf error norms for rho, u, and p.

    The discrete error norms are defined as (see docs/08_test_cases.md, Section 8.6):

        L1   = sum(|f_i - f_exact_i|) * dx
        L2   = sqrt(sum((f_i - f_exact_i)**2) * dx)
        Linf = max(|f_i - f_exact_i|)

    These correspond to standard quadrature approximations of the continuous
    L1, L2, and Linf norms on the computational domain.

    Parameters
    ----------
    result : SolverResult
        Numerical solution (final snapshot is used).
    exact : pd.DataFrame
        Exact solution on the same mesh, as returned by
        :func:`compute_exact_solution`.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict ``{variable: {norm: value}}``, e.g.
        ``errors["rho"]["L2"]``.
    """
    df = result_to_dataframe(result)
    dx = result.config.mesh.dx
    errors: dict[str, dict[str, float]] = {}

    for var in ("rho", "u", "p"):
        diff = np.abs(df[var].values - exact[var].values)
        errors[var] = {
            "L1": float(np.sum(diff) * dx),
            "L2": float(np.sqrt(np.sum(diff**2) * dx)),
            "Linf": float(np.max(diff)),
        }

    return errors


def results_to_comparison_df(
    results: list[SolverResult], config: SimulationConfig
) -> pd.DataFrame:
    """Merge multiple scheme results into a long-format DataFrame for plotting.

    Converts each :class:`SolverResult` to a DataFrame, tags it with the
    scheme name, appends the exact solution (tagged ``"Exact"``), and
    concatenates everything into a single long-format DataFrame suitable
    for grouped Plotly/Matplotlib plots.

    Parameters
    ----------
    results : list[SolverResult]
        One result per scheme to compare.
    config : SimulationConfig
        Used to compute the exact reference solution.

    Returns
    -------
    pd.DataFrame
        Long-format with an extra ``scheme`` column.
    """
    frames = []
    for r in results:
        df = result_to_dataframe(r)
        df["scheme"] = r.scheme_name
        frames.append(df)

    exact = compute_exact_solution(config)
    exact["scheme"] = "Exact"
    frames.append(exact)

    return pd.concat(frames, ignore_index=True)


def convergence_study(
    make_config: Callable[[int], SimulationConfig],
    scheme: "NumericalScheme",
    n_cells_list: list[int],
    variables: tuple[str, ...] = ("rho", "u", "p"),
    time_integrator: str | None = None,
) -> pd.DataFrame:
    """Run a mesh convergence study at constant CFL (Section 8.6).

    For each resolution in *n_cells_list*, this function:
        1. Builds a ``SimulationConfig`` via *make_config(n_cells)*.
        2. Computes the exact solution on that mesh.
        3. Runs the simulation with the given *scheme*.
        4. Computes L1, L2, Linf error norms for each requested variable.

    The CFL number is kept constant across resolutions (set inside
    *make_config*), so that dt scales proportionally to dx and the
    temporal error remains of the same order as the spatial error.

    The resulting DataFrame can be fed to :func:`estimate_order` to
    extract the convergence rate via log-log regression.

    Parameters
    ----------
    make_config : callable
        Factory ``(n_cells: int) -> SimulationConfig``.  Must keep CFL
        constant across resolutions.
    scheme : NumericalScheme
        The numerical scheme to evaluate.
    n_cells_list : list of int
        Mesh resolutions to test (e.g. ``[50, 100, 200, 400, 800]``).
    variables : tuple of str
        Variables on which to compute errors (default ``("rho", "u", "p")``).
    time_integrator : str or None
        Override the time integrator (default: scheme's default).

    Returns
    -------
    pd.DataFrame
        Columns: ``n_cells, dx, variable, L1, L2, Linf, scheme``.
    """
    rows: list[dict] = []
    for n in n_cells_list:
        config = make_config(n)
        exact = compute_exact_solution(config)
        result = run_simulation(config, scheme, time_integrator=time_integrator)
        if not result.snapshots:
            continue
        errs = compute_errors(result, exact)
        for var in variables:
            rows.append({
                "n_cells": n,
                "dx": config.mesh.dx,
                "variable": var,
                "L1": errs[var]["L1"],
                "L2": errs[var]["L2"],
                "Linf": errs[var]["Linf"],
                "scheme": scheme.name,
            })
    return pd.DataFrame(rows)


def estimate_order(dx_values: np.ndarray, error_values: np.ndarray) -> float:
    """Estimate convergence order via least-squares fit in log-log space (Section 8.6).

    Fits the linear model ``log(err) = p * log(dx) + c`` using numpy's
    :func:`numpy.polyfit`.  The slope *p* is the estimated convergence order.

    Expected values on smooth test cases: p ~ 1 for first-order schemes,
    p ~ 2 for MUSCL/ENO2/Lax-Wendroff, p ~ 3 for WENO3, p ~ 5 for WENO5.

    Parameters
    ----------
    dx_values : array-like
        Mesh spacings (must be strictly positive).
    error_values : array-like
        Corresponding error norms (must be strictly positive).

    Returns
    -------
    float
        Estimated convergence order *p*.
    """
    log_dx = np.log(dx_values)
    log_err = np.log(error_values)
    p, _ = np.polyfit(log_dx, log_err, 1)
    return float(p)
