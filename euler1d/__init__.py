"""Euler 1D solver package.

Numerical solver for the one-dimensional Euler equations of compressible gas
dynamics using finite volume methods. Designed for comparing numerical schemes
(Godunov, HLL, HLLC, Roe, Rusanov, Lax-Friedrichs, Lax-Wendroff, JST) with
high-order reconstructions (MUSCL, ENO, WENO) on classical Riemann problems
and smooth test cases.

Architecture overview
---------------------
- **config** : Problem definition (gas properties, mesh, time, initial conditions).
- **physics** : Equation of state, primitive/conservative variable conversions,
  physical flux computation.
- **schemes/** : Numerical flux functions (flux/) and spatial reconstructions
  (reconstruction/). Composed via the "reconstruction-flux" naming convention.
- **solver** : Time-stepping loop with explicit Runge-Kutta integration.
- **boundary** : Ghost-cell boundary conditions (transmissive, reflective, periodic).
- **riemann** : Exact Riemann solver for reference solutions.
- **test_cases** : Predefined problems (Sod, Lax, double rarefaction, entropy wave,
  acoustic wave).
- **results** : Error norms, convergence studies, exact solution computation.
- **fourier_analysis** : Numerical dissipation and dispersion analysis.

Mathematical background
-----------------------
The 1D Euler equations in conservative form are dU/dt + dF(U)/dx = 0 where
U = (rho, rho*u, E) and F(U) = (rho*u, rho*u^2+p, u*(E+p)). The system is
closed by the perfect gas equation of state p = (gamma-1)*(E - 0.5*rho*u^2).

See Also
--------
docs/01_euler_equations.md : Chapter 1 — full derivation of the Euler equations,
    equation of state, hyperbolicity, and wave structure.

References
----------
[Toro, 2009] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid
    Dynamics, 3rd ed., Springer.
[LeVeque, 2002] R.J. LeVeque, Finite Volume Methods for Hyperbolic Problems,
    Cambridge University Press.
"""
