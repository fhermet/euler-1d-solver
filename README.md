# Solveur Euler 1D

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://euler-1d-solver.streamlit.app/)

Solveur modulaire pour les equations d'Euler 1D instationnaires (gaz parfait), ecrit en Python/NumPy. Interface interactive Streamlit pour comparer 59 schemas de volumes finis sur des problemes de Riemann classiques et des solutions lisses, avec etude de convergence et analyse de Fourier.

**Essayer en ligne** : https://euler-1d-solver.streamlit.app/

## Prerequis

- Python >= 3.10
- NumPy, Pandas, Matplotlib, Plotly, Streamlit

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run Solveur_Euler_1D.py
```

## Architecture du code

```
Solveur_Euler_1D.py    Point d'entree Streamlit (page principale)
ui_common.py           Style Plotly, constantes, selecteurs de schemas partages
pages/
└── 2_Analyse_de_Fourier.py   Page Streamlit : dissipation et dispersion numerique

euler1d/
├── config.py          Dataclasses de configuration (gaz, maillage, temps, probleme)
├── physics.py         EOS, conversions primitif/conservatif, flux physique
├── riemann.py         Solveur de Riemann exact (vectorise numpy)
├── boundary.py        Conditions aux limites (cellules fantomes)
├── solver.py          Boucle en temps (RK1 a RK5)
├── results.py         Solution exacte, normes d'erreur, convergence
├── fourier_analysis.py  Analyse de dissipation/dispersion numerique
├── test_cases.py      Cas tests predefinis
└── schemes/
    ├── base.py              Classe abstraite NumericalScheme
    ├── flux/
    │   ├── rusanov.py         Rusanov (Lax-Friedrichs local)
    │   ├── hll.py             HLL (2 ondes)
    │   ├── hllc.py            HLLC (3 ondes)
    │   ├── roe.py             Roe avec correction entropique
    │   ├── godunov.py         Godunov (solveur exact)
    │   ├── ausm_plus.py       AUSM+ et AUSM+-up
    │   ├── lax_friedrichs.py  Lax-Friedrichs (centre)
    │   ├── lax_wendroff.py    Lax-Wendroff (centre)
    │   └── jst.py             JST (centre)
    └── reconstruction/
        ├── limiters.py    Limiteurs de pente (minmod, van-leer, superbee, mc, van-albada)
        ├── muscl.py       Reconstruction MUSCL (ordre 2)
        ├── eno.py         Reconstruction ENO2 (ordre 2)
        ├── weno3.py       WENO3 JS et Z (ordre 3)
        └── weno5.py       WENO5 JS et Z (ordre 5)
```

## Schemas disponibles

### Flux numeriques (ordre 1)

| Schema | Cle | Description |
|---|---|---|
| Lax-Friedrichs | `lax-friedrichs` | Dissipation globale, schema centre |
| Rusanov | `rusanov` | Dissipation locale (Lax-Friedrichs local) |
| HLL | `hll` | 2 ondes, pas de contact |
| HLLC | `hllc` | 3 ondes, capture le contact |
| Roe | `roe` | Linearisation + correction entropique |
| Roe (sans correction) | `roe-nc` | Roe sans correction entropique |
| Godunov | `godunov` | Solveur de Riemann exact |
| AUSM+ | `ausm+` | Splitting Mach/pression |
| AUSM+-up | `ausm+-up` | AUSM+ ameliore (bas Mach) |
| Lax-Wendroff | `lax-wendroff` | Ordre 2 en temps et espace, schema centre |
| JST | `jst` | Jameson-Schmidt-Turkel, schema centre |

### Reconstructions d'ordre eleve

| Reconstruction | Cle | Ordre | Composable avec |
|---|---|---|---|
| MUSCL | `muscl-{flux}` | 2 | Rusanov, HLL, HLLC, Roe, Roe-NC, Godunov, AUSM+, AUSM+-up |
| ENO2 | `eno2-{flux}` | 2 | idem |
| WENO3-JS | `weno3-{flux}` | 3 | idem |
| WENO3-Z | `wenoz3-{flux}` | 3 | idem |
| WENO5-JS | `weno5-{flux}` | 5 | idem |
| WENO5-Z | `wenoz5-{flux}` | 5 | idem |

8 flux composables x 6 reconstructions + 11 schemas autonomes = **59 schemas** au total.

Limiteurs MUSCL : `minmod`, `van-leer` (defaut), `superbee`, `mc`, `van-albada`.

## Interface Streamlit

### Page principale : Solveur Euler 1D

| Onglet | Description |
|---|---|
| Ordre de convergence | Profils (rho, u, p) + solution exacte, erreurs L1/L2/Linf, convergence en maillage |
| Evolution temporelle | Animation dans le temps avec slider, solution exacte a t arbitraire |

### Page : Analyse de Fourier

Courbes de dissipation |G(theta)| et dispersion phi/phi_exact pour chaque schema, obtenues par linearisation autour d'un etat uniforme.

## Cas tests disponibles

| Cas | Fonction | Type | Description |
|---|---|---|---|
| Sod | `sod_shock_tube()` | Riemann | Choc + contact + rarefaction |
| Lax | `lax_test()` | Riemann | Plus severe que Sod |
| Double rarefaction | `double_rarefaction()` | Riemann | Deux rarefactions, basse pression |
| Contact stationnaire | `stationary_contact()` | Riemann | Contact immobile, mesure de diffusion |
| Quasi-vide | `near_vacuum()` | Riemann | Double detente extreme (93% du seuil de vide) |
| Deux chocs | `two_shocks()` | Riemann | Collision de chocs forts (Toro test 4) |
| Shu-Osher | `shu_osher()` | Mixte | Choc Mach 3 dans densite sinusoidale |
| Onde d'entropie | `entropy_wave()` | Lisse | Advection sinusoidale, verification d'ordre |
| Onde acoustique | `acoustic_wave()` | Lisse | Perturbation isentropique, verification d'ordre |

## Utilisation en tant que bibliotheque

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

## Documentation theorique

La documentation mathematique detaillee est dans le dossier [`docs/`](docs/) :

1. [Equations d'Euler 1D](docs/01_euler_equations.md)
2. [Probleme de Riemann](docs/02_riemann_problem.md)
3. [Methode des volumes finis](docs/03_finite_volume.md)
4. [Schemas de flux](docs/04_flux_schemes.md)
5. [Reconstructions d'ordre eleve](docs/05_reconstruction.md)
6. [Integration temporelle](docs/06_time_integration.md)
7. [Analyse de Fourier](docs/07_fourier_analysis.md)
8. [Cas tests](docs/08_test_cases.md)
9. [Bibliographie](docs/bibliography.md)

## References

1. E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd Edition, Springer, 2009.
2. B. van Leer, "Towards the ultimate conservative difference scheme. V.", *J. Comput. Phys.*, 32, 1979.
3. P. L. Roe, "Approximate Riemann solvers", *J. Comput. Phys.*, 43, 1981.
4. G.-S. Jiang, C.-W. Shu, "Efficient implementation of weighted ENO schemes", *J. Comput. Phys.*, 126, 1996.
