# Solveur Euler 1D

Solveur modulaire pour les equations d'Euler 1D instationnaires (gaz parfait), ecrit en Python/NumPy. Implementations de 6 solveurs de Riemann a l'ordre 1 et 5 schemas MUSCL a l'ordre 2, avec etude de convergence et comparaison de limiteurs.

## Prerequis

- Python >= 3.10
- NumPy
- Matplotlib
- Pandas

```bash
pip install numpy matplotlib pandas
```

## Architecture du code

```
euler1d/
├── config.py          Dataclasses de configuration (gaz, maillage, temps, probleme)
├── physics.py         EOS, conversions primitif/conservatif, flux physique
├── riemann.py         Solveur de Riemann exact (vectorise numpy)
├── boundary.py        Conditions aux limites (cellules fantomes)
├── solver.py          Boucle en temps (Euler explicite / RK2)
├── results.py         Export DataFrame, solution exacte, normes d'erreur
├── test_cases.py      Cas tests predefinis (Sod, Lax, double rarefaction)
└── schemes/
    ├── base.py              Classe abstraite NumericalScheme
    ├── reconstruction.py    Reconstruction MUSCL + limiteurs de pente
    ├── order1/
    │   ├── lax_friedrichs.py  Schema de Lax-Friedrichs
    │   ├── rusanov.py         Schema de Rusanov
    │   ├── hll.py             Schema HLL
    │   ├── hllc.py            Schema HLLC
    │   ├── roe.py             Schema de Roe avec correction entropique
    │   └── godunov.py         Schema de Godunov (solveur exact)
    └── order2/
        ├── muscl_rusanov.py   MUSCL-Rusanov
        ├── muscl_hll.py       MUSCL-HLL
        ├── muscl_hllc.py      MUSCL-HLLC
        ├── muscl_roe.py       MUSCL-Roe
        └── muscl_godunov.py   MUSCL-Godunov
```

## Schemas disponibles

### Ordre 1

| Schema | Cle | Description |
|---|---|---|
| Lax-Friedrichs | `lax-friedrichs` | Dissipation globale, le plus simple |
| Rusanov | `rusanov` | Dissipation locale (Lax-Friedrichs local) |
| HLL | `hll` | 2 ondes, pas de contact |
| HLLC | `hllc` | 3 ondes, capture le contact |
| Roe | `roe` | Linearisation + correction entropique |
| Godunov | `godunov` | Solveur de Riemann exact |

### Ordre 2 (MUSCL + RK2)

| Schema | Cle | Solveur de Riemann |
|---|---|---|
| MUSCL-Rusanov | `muscl-rusanov` | Rusanov |
| MUSCL-HLL | `muscl-hll` | HLL |
| MUSCL-HLLC | `muscl-hllc` | HLLC |
| MUSCL-Roe | `muscl-roe` | Roe |
| MUSCL-Godunov | `muscl-godunov` | Godunov (exact) |

Limiteurs disponibles : `minmod`, `van-leer` (defaut), `superbee`, `mc`, `van-albada`.

## Utilisation

### Scripts en ligne de commande

**Comparaison de tous les schemas sur le tube de Sod** :

```bash
python3 run.py
```

Produit `sod_comparison.png` avec les profils de densite, vitesse et pression.

**Etude de convergence en maillage** :

```bash
python3 convergence.py                                # tous les schemas, variable rho (Sod)
python3 convergence.py -t entropy_wave                # onde d'entropie (solution lisse)
python3 convergence.py -t acoustic_wave               # onde acoustique (solution lisse)
python3 convergence.py -t entropy_wave -s rusanov hllc muscl-hllc  # sous-ensemble sur cas lisse
python3 convergence.py --order 2                      # schemas ordre 2 uniquement
python3 convergence.py -s hllc roe muscl-hllc         # sous-ensemble de schemas
python3 convergence.py -s muscl-hllc -l superbee      # MUSCL-HLLC avec limiteur superbee
python3 convergence.py -v u                           # convergence sur la vitesse
python3 convergence.py -n 50 100 200 400 800 1600     # resolutions personnalisees
python3 convergence.py -o convergence_rho.png         # nom du fichier de sortie
```

**Comparaison des limiteurs de pente** :

```bash
python3 compare_limiters.py                           # defaut : muscl-hllc
python3 compare_limiters.py -s muscl-roe              # schema MUSCL au choix
```

### Utilisation en tant que bibliotheque

```python
from euler1d.schemes import get_scheme
from euler1d.solver import run_simulation
from euler1d.test_cases import sod_shock_tube
from euler1d.results import compute_errors, compute_exact_solution, result_to_dataframe

# Configurer et lancer une simulation
config = sod_shock_tube(n_cells=200)
scheme = get_scheme("muscl-hllc", limiter="van_leer")
result = run_simulation(config, scheme)

# Exploiter les resultats
df = result_to_dataframe(result)
exact = compute_exact_solution(config)
errors = compute_errors(result, exact)
print(errors["rho"])  # {'L1': ..., 'L2': ..., 'Linf': ...}
```

## Cas tests disponibles

| Cas | Fonction | Description |
|---|---|---|
| Sod | `sod_shock_tube()` | Tube a choc classique. Choc droit, contact, rarefaction gauche. |
| Lax | `lax_test()` | Probleme de Lax. Ondes plus fortes que Sod. |
| Double rarefaction | `double_rarefaction()` | Probleme 123. Deux rarefactions symetriques, creation de vide. |
| Onde d'entropie | `entropy_wave()` | Perturbation sinusoidale de densite advectee (CL periodiques). Solution lisse pour verification d'ordre. |
| Onde acoustique | `acoustic_wave()` | Petite perturbation isentropique (CL periodiques). Solution linearisee pour verification d'ordre. |

## Ajouter un nouveau schema

1. Creer un fichier dans `euler1d/schemes/order1/` ou `order2/`
2. Heriter de `NumericalScheme`, implementer `name` et `compute_riemann_flux`
3. Pour l'ordre 2 : definir `order = 2`, `n_ghost = 2`, surcharger `compute_fluxes` avec reconstruction MUSCL
4. Ajouter une ligne dans `SCHEME_REGISTRY` de `schemes/__init__.py`

## Documentation theorique

La documentation mathematique et physique detaillee est dans le dossier [`docs/`](docs/) :

1. [Equations d'Euler 1D](docs/01_equations_euler.md) -- systeme d'equations, EOS, variables
2. [Methode des volumes finis](docs/02_volumes_finis.md) -- discretisation, flux numerique, CFL
3. [Solveurs de Riemann](docs/03_solveurs_riemann.md) -- les 6 solveurs avec formules et comparaison
4. [Reconstruction MUSCL](docs/04_reconstruction_muscl.md) -- MUSCL, limiteurs de pente, region de Sweby
5. [Integration temporelle](docs/05_integration_temporelle.md) -- Euler explicite, RK2, alternatives
6. [Ordre et convergence](docs/06_ordre_et_convergence.md) -- normes d'erreur, analyse de convergence
7. [Extensions](docs/07_extensions.md) -- PPM, WENO, DG, idees recues sur l'ordre

## Interface graphique (Streamlit)

Une interface interactive Streamlit unifie toutes les fonctionnalites dans un navigateur web.

### Installation

```bash
pip install streamlit plotly
```

### Lancement

```bash
streamlit run app.py
```

### Fonctionnalites

| Onglet | Description |
|---|---|
| Comparaison de schemas | Profils (rho, u, p) + solution exacte, tableau d'erreurs L1/L2/Linf |
| Etude de convergence | Graphiques log-log avec pentes de reference, ordres estimes |
| Comparaison de limiteurs | Profils et convergence par limiteur, reference ordre 1 optionnelle |
| Animation temporelle | Navigation dans le temps avec slider, solution exacte a t arbitraire |

La barre laterale permet de choisir le cas test, le nombre de cellules, le CFL et le gamma. Les calculs sont caches pour une navigation fluide.

## References

1. E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd Edition, Springer, 2009.
2. B. van Leer, "Towards the ultimate conservative difference scheme. V.", *J. Comput. Phys.*, 32, 1979.
3. P. L. Roe, "Approximate Riemann solvers", *J. Comput. Phys.*, 43, 1981.
