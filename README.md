# 1D Euler Solver

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://euler-1d-solver.streamlit.app/)

Modular solver for the unsteady 1D Euler equations (ideal gas), written in Python/NumPy. Interactive Streamlit interface to compare 59 finite volume schemes on classical Riemann problems and smooth solutions, with mesh convergence studies and Fourier analysis.

**Try it online**: https://euler-1d-solver.streamlit.app/

## Prerequisites

- Python >= 3.10
- NumPy, Pandas, Matplotlib, Plotly, Streamlit

```bash
pip install -r requirements.txt
```

## Getting started

```bash
streamlit run Euler_1D_Solver.py
```

## Code architecture

```
Euler_1D_Solver.py     Streamlit entry point (main page)
ui_common.py           Plotly styling, constants, shared scheme selectors
pages/
└── 2_Fourier_Analysis.py   Streamlit page: numerical dissipation and dispersion

euler1d/
├── config.py          Configuration dataclasses (gas, mesh, time, problem)
├── physics.py         EOS, primitive/conservative conversions, physical flux
├── riemann.py         Exact Riemann solver (vectorized NumPy)
├── boundary.py        Boundary conditions (ghost cells)
├── solver.py          Time-stepping loop (RK1 to RK5)
├── results.py         Exact solution, error norms, convergence
├── fourier_analysis.py  Numerical dissipation/dispersion analysis
├── test_cases.py      Predefined test cases
└── schemes/
    ├── base.py              Abstract base class NumericalScheme
    ├── flux/
    │   ├── rusanov.py         Rusanov (local Lax-Friedrichs)
    │   ├── hll.py             HLL (2-wave)
    │   ├── hllc.py            HLLC (3-wave)
    │   ├── roe.py             Roe with entropy fix
    │   ├── godunov.py         Godunov (exact solver)
    │   ├── ausm_plus.py       AUSM+ and AUSM+-up
    │   ├── lax_friedrichs.py  Lax-Friedrichs (centered)
    │   ├── lax_wendroff.py    Lax-Wendroff (centered)
    │   └── jst.py             JST (centered)
    └── reconstruction/
        ├── limiters.py    Slope limiters (minmod, van-leer, superbee, mc, van-albada)
        ├── muscl.py       MUSCL reconstruction (order 2)
        ├── eno.py         ENO2 reconstruction (order 2)
        ├── weno3.py       WENO3 JS and Z (order 3)
        └── weno5.py       WENO5 JS and Z (order 5)
```

## Available schemes

### Numerical fluxes (order 1)

| Scheme | Key | Description |
|---|---|---|
| Lax-Friedrichs | `lax-friedrichs` | Global dissipation, centered scheme |
| Rusanov | `rusanov` | Local dissipation (local Lax-Friedrichs) |
| HLL | `hll` | 2-wave, no contact resolution |
| HLLC | `hllc` | 3-wave, captures the contact |
| Roe | `roe` | Linearization + entropy fix |
| Roe (no correction) | `roe-nc` | Roe without entropy fix |
| Godunov | `godunov` | Exact Riemann solver |
| AUSM+ | `ausm+` | Mach/pressure splitting |
| AUSM+-up | `ausm+-up` | Improved AUSM+ (low Mach) |
| Lax-Wendroff | `lax-wendroff` | 2nd order in time and space, centered scheme |
| JST | `jst` | Jameson-Schmidt-Turkel, centered scheme |

### High-order reconstructions

| Reconstruction | Key | Order | Composable with |
|---|---|---|---|
| MUSCL | `muscl-{flux}` | 2 | Rusanov, HLL, HLLC, Roe, Roe-NC, Godunov, AUSM+, AUSM+-up |
| ENO2 | `eno2-{flux}` | 2 | same |
| WENO3-JS | `weno3-{flux}` | 3 | same |
| WENO3-Z | `wenoz3-{flux}` | 3 | same |
| WENO5-JS | `weno5-{flux}` | 5 | same |
| WENO5-Z | `wenoz5-{flux}` | 5 | same |

8 composable fluxes × 6 reconstructions + 11 standalone schemes = **59 schemes** total.

MUSCL limiters: `minmod`, `van-leer` (default), `superbee`, `mc`, `van-albada`.

## Streamlit interface

### Main page: 1D Euler Solver

| Tab | Description |
|---|---|
| Convergence order | Profiles (ρ, u, p) + exact solution, L1/L2/L∞ errors, mesh convergence |
| Time evolution | Time animation with slider, exact solution at arbitrary t |

### Page: Fourier Analysis

Dissipation |G(θ)| and dispersion φ/φ_exact curves for each scheme, obtained by linearization around a uniform state.

## Available test cases

| Case | Function | Type | Description |
|---|---|---|---|
| Sod | `sod_shock_tube()` | Riemann | Shock + contact + rarefaction |
| Lax | `lax_test()` | Riemann | More severe than Sod |
| Double rarefaction | `double_rarefaction()` | Riemann | Two rarefactions, low pressure |
| Stationary contact | `stationary_contact()` | Riemann | Stationary contact, diffusion measurement |
| Near vacuum | `near_vacuum()` | Riemann | Extreme double rarefaction (93% of vacuum threshold) |
| Two shocks | `two_shocks()` | Riemann | Strong shock collision (Toro test 4) |
| Shu-Osher | `shu_osher()` | Mixed | Mach 3 shock in sinusoidal density |
| Entropy wave | `entropy_wave()` | Smooth | Sinusoidal advection, order verification |
| Acoustic wave | `acoustic_wave()` | Smooth | Isentropic perturbation, order verification |

## Usage as a library

```python
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import sod_shock_tube
from euler1d.results import compute_errors, compute_exact_solution, result_to_dataframe

config = sod_shock_tube(n_cells=200)
scheme = get_scheme("muscl-hllc", limiter="van-leer")
result = run_simulation(config, scheme)

df = result_to_dataframe(result)
exact = compute_exact_solution(config)
errors = compute_errors(result, exact)
print(errors["rho"])  # {'L1': ..., 'L2': ..., 'Linf': ...}
```

## Theoretical documentation

Detailed mathematical documentation is in the [`docs/`](docs/) folder:

1. [1D Euler equations](docs/01_euler_equations.md)
2. [Riemann problem](docs/02_riemann_problem.md)
3. [Finite volume method](docs/03_finite_volume.md)
4. [Flux schemes](docs/04_flux_schemes.md)
5. [High-order reconstructions](docs/05_reconstruction.md)
6. [Time integration](docs/06_time_integration.md)
7. [Fourier analysis](docs/07_fourier_analysis.md)
8. [Test cases](docs/08_test_cases.md)
9. [Bibliography](docs/bibliography.md)

## References

1. E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd Edition, Springer, 2009.
2. B. van Leer, "Towards the ultimate conservative difference scheme. V.", *J. Comput. Phys.*, 32, 1979.
3. P. L. Roe, "Approximate Riemann solvers", *J. Comput. Phys.*, 43, 1981.
4. G.-S. Jiang, C.-W. Shu, "Efficient implementation of weighted ENO schemes", *J. Comput. Phys.*, 126, 1996.
