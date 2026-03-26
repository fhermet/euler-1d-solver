# Chapitre 3 : Méthode des volumes finis

> **Prérequis** : [Chapitre 1 — Équations d'Euler](01_euler_equations.md) (système hyperbolique, variables conservatives et primitives), [Chapitre 2 — Problème de Riemann](02_riemann_problem.md) (structure des ondes, solutions faibles).

## Introduction

Les deux premiers chapitres ont établi le cadre physique : les équations d'Euler 1D forment un système hyperbolique de lois de conservation, et le problème de Riemann en est la brique élémentaire. Nous savons caractériser analytiquement les ondes (détentes, contacts, chocs) qui naissent d'une discontinuité initiale.

Le passage au calcul numérique pose une question fondamentale : comment transformer l'EDP continue

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0$$

en un algorithme évaluable par un ordinateur, tout en respectant la structure physique du problème — en particulier la conservativité et la capacité à capturer les discontinuités ?

La **méthode des volumes finis** répond à cette question en partant de la forme intégrale des lois de conservation. Contrairement aux différences finies qui discrétisent directement les dérivées, les volumes finis discrétisent les bilans de flux à travers les frontières de cellules. Cette approche est intrinsèquement conservative : ce qui sort d'une cellule entre dans la voisine.

Ce chapitre construit la méthode étape par étape, depuis la formulation intégrale exacte jusqu'à l'algorithme complet implémenté dans le code.

---

## §3.1 Formulation intégrale

### Du continu au discret : le maillage

On subdivise le domaine $[a, b]$ en $N$ cellules de taille uniforme $\Delta x = (b-a)/N$. La cellule $i$ est l'intervalle $[x_{i-1/2},\, x_{i+1/2}]$, de centre $x_i = \tfrac{1}{2}(x_{i-1/2} + x_{i+1/2})$.

### Intégration sur une cellule

Intégrons l'équation d'Euler sur la cellule $i$ :

$$\int_{x_{i-1/2}}^{x_{i+1/2}} \frac{\partial \mathbf{U}}{\partial t}\, dx + \int_{x_{i-1/2}}^{x_{i+1/2}} \frac{\partial \mathbf{F}}{\partial x}\, dx = 0$$

Le second terme se simplifie exactement par le théorème fondamental de l'analyse :

$$\int_{x_{i-1/2}}^{x_{i+1/2}} \frac{\partial \mathbf{F}}{\partial x}\, dx = \mathbf{F}\bigl(\mathbf{U}(x_{i+1/2}, t)\bigr) - \mathbf{F}\bigl(\mathbf{U}(x_{i-1/2}, t)\bigr)$$

### Moyenne de cellule

On définit la **moyenne de cellule** :

$$\bar{\mathbf{U}}_i(t) = \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \mathbf{U}(x,t)\, dx$$

En divisant l'équation intégrée par $\Delta x$, on obtient :

$$\frac{d\bar{\mathbf{U}}_i}{dt} = -\frac{1}{\Delta x} \Bigl[ \mathbf{F}\bigl(\mathbf{U}(x_{i+1/2}, t)\bigr) - \mathbf{F}\bigl(\mathbf{U}(x_{i-1/2}, t)\bigr) \Bigr]$$

**Point essentiel** : cette équation est **exacte**. Aucune approximation n'a été faite. L'évolution de la moyenne de cellule est entièrement déterminée par les flux physiques aux interfaces. La difficulté est que nous ne connaissons pas la valeur ponctuelle $\mathbf{U}(x_{i+1/2}, t)$ — nous ne disposons que des moyennes $\bar{\mathbf{U}}_i$.

---

## §3.2 Flux numérique

### Le problème

À l'interface $x_{i+1/2}$, la solution $\mathbf{U}$ n'est pas connue. Nous disposons uniquement des moyennes de cellule $\bar{\mathbf{U}}_i$ et $\bar{\mathbf{U}}_{i+1}$ de part et d'autre. Il faut **approximer** le flux physique $\mathbf{F}(\mathbf{U}(x_{i+1/2}))$ par un **flux numérique** :

$$\hat{\mathbf{F}}_{i+1/2} = \hat{\mathbf{F}}(\mathbf{U}_L,\, \mathbf{U}_R)$$

où $\mathbf{U}_L$ et $\mathbf{U}_R$ sont des estimations des états de part et d'autre de l'interface (voir §3.4 pour leur construction).

C'est ici que le problème de Riemann du chapitre 2 intervient : chaque interface est vue comme un mini-problème de Riemann entre l'état gauche $\mathbf{U}_L$ et l'état droit $\mathbf{U}_R$.

### Propriétés requises

Un flux numérique doit satisfaire trois propriétés fondamentales :

1. **Consistance** : si les deux états sont identiques, le flux numérique redonne le flux physique :
$$\hat{\mathbf{F}}(\mathbf{U}, \mathbf{U}) = \mathbf{F}(\mathbf{U})$$

2. **Conservativité** : le flux $\hat{\mathbf{F}}_{i+1/2}$ est unique à chaque interface. Ce qui sort de la cellule $i$ par sa frontière droite est exactement ce qui entre dans la cellule $i+1$ par sa frontière gauche. Le **théorème de Lax-Wendroff** garantit que si un schéma conservatif converge, il converge vers une solution faible des équations de conservation. Sans conservativité, un schéma peut converger vers une solution qui ne respecte pas les relations de saut (Rankine-Hugoniot).

3. **Continuité Lipschitz** : $\hat{\mathbf{F}}$ est Lipschitz-continue par rapport à ses arguments, ce qui assure la stabilité du schéma.

### Les différents solveurs de Riemann approchés

Les chapitres 4 détailleront les flux numériques implémentés dans le code : Rusanov, HLL, HLLC, Roe, Godunov (solveur exact), ainsi que les schémas centrés (Lax-Friedrichs, Lax-Wendroff, JST). Chacun réalise un compromis différent entre précision, coût de calcul et robustesse.

---

## §3.3 Semi-discrétisation

### Méthode des lignes

En remplaçant les flux exacts par les flux numériques, on obtient la **semi-discrétisation** :

$$\frac{d\bar{\mathbf{U}}_i}{dt} = -\frac{1}{\Delta x} \Bigl( \hat{\mathbf{F}}_{i+1/2} - \hat{\mathbf{F}}_{i-1/2} \Bigr) \quad \text{pour } i = 1, \dots, N$$

C'est un **système d'EDO** (équations différentielles ordinaires) : la variable continue est le temps $t$, et l'espace a été discrétisé. Cette approche est appelée **méthode des lignes** (*method of lines*).

### Intégration temporelle

Le système d'EDO est de la forme :

$$\frac{d\bar{\mathbf{U}}}{dt} = \mathbf{L}(\bar{\mathbf{U}})$$

où $\mathbf{L}$ est l'opérateur spatial (calcul des flux numériques). On avance en temps par une méthode de **Runge-Kutta** explicite. Le choix de la méthode RK dépend de l'ordre spatial du schéma — le chapitre 6 détaillera les intégrateurs temporels (RK1 à RK5) et leur couplage avec les schémas spatiaux.

### Pas de temps adaptatif

Le pas de temps est contraint par la condition **CFL** (Courant-Friedrichs-Lewy) :

$$\Delta t = \text{CFL} \cdot \frac{\Delta x}{\max_i (\lvert u_i \rvert + c_i)}$$

où $u_i$ est la vitesse locale et $c_i$ la vitesse du son. La constante CFL doit être inférieure ou égale à 1 pour la stabilité (typiquement CFL $= 0.5$ à $0.9$). Le dénominateur est la vitesse maximale de propagation de l'information dans le domaine, calculée à chaque pas de temps.

---

## §3.4 Reconstruction

### Ordre 1 : reconstruction constante par morceaux

Au premier ordre, on prend simplement :

$$\mathbf{U}_L = \bar{\mathbf{U}}_i, \qquad \mathbf{U}_R = \bar{\mathbf{U}}_{i+1}$$

Chaque cellule est représentée par une valeur constante égale à sa moyenne. Cette reconstruction est stable mais très diffusive : les discontinuités sont étalées sur de nombreuses cellules.

### Ordre élevé : de la moyenne à l'interface

Pour améliorer la précision, on **reconstruit** les valeurs aux interfaces à partir des moyennes de cellule voisines. L'idée est d'utiliser un polynôme interpolant les moyennes sur un stencil (groupe de cellules voisines) pour estimer $\mathbf{U}(x_{i+1/2})$ plus précisément.

Par exemple :
- **MUSCL** (ordre 2) : reconstruction linéaire avec limiteur de pente pour éviter les oscillations (chapitre 5).
- **ENO** (ordre 2) : sélection adaptative du stencil le plus régulier.
- **WENO** (ordres 3 et 5) : combinaison pondérée de plusieurs stencils, avec des poids qui s'adaptent à la régularité locale de la solution.

### Pourquoi les variables primitives ?

**Convention importante** : toutes les reconstructions sont effectuées sur les **variables primitives** $\mathbf{W} = (\rho, u, p)$, et non sur les variables conservatives $\mathbf{U} = (\rho, \rho u, E)$. Les variables primitives sont mieux conditionnées pour l'interpolation au voisinage des discontinuités : reconstruire l'énergie totale $E$ (qui mélange effets cinétiques et thermodynamiques) produit des oscillations parasites que la reconstruction sur $(\rho, u, p)$ évite.

### Lien entre reconstruction et ordre de convergence

L'ordre spatial du schéma global est déterminé par l'ordre de la reconstruction (à condition que l'intégrateur temporel soit d'ordre au moins égal). Une reconstruction d'ordre $k$ signifie que l'erreur de troncature spatiale est en $O(\Delta x^k)$ en zone lisse. Le chapitre 5 développe la théorie et l'implémentation des reconstructions.

---

## §3.5 Principe de composition

### Schéma = reconstruction + flux numérique

Un schéma de volumes finis d'ordre élevé se décompose en deux briques indépendantes :

1. **Reconstruction** : à partir des moyennes de cellule, estimer les états $\mathbf{U}_L$ et $\mathbf{U}_R$ aux interfaces.
2. **Flux numérique** : à partir de $\mathbf{U}_L$ et $\mathbf{U}_R$, calculer le flux $\hat{\mathbf{F}}_{i+1/2}$.

Ces deux briques sont **interchangeables** : on peut combiner n'importe quelle reconstruction (constante, MUSCL, ENO, WENO3, WENO5) avec n'importe quel solveur de Riemann (Rusanov, HLL, HLLC, Roe, Godunov). Cela donne une grande variété de schémas à partir d'un petit nombre de composants.

### Interface abstraite

Dans le code, cette composition est réalisée par la classe abstraite `NumericalScheme` (fichier `euler1d/schemes/base.py`). L'algorithme est le suivant :

1. `compute_fluxes(U, gas, dx, dt)` est appelé avec le tableau étendu (cellules intérieures + fantômes).
2. Par défaut (ordre 1), il extrait $\mathbf{U}_L = \mathbf{U}_{:, i}$ et $\mathbf{U}_R = \mathbf{U}_{:, i+1}$ (reconstruction constante).
3. Il appelle `compute_riemann_flux(UL, UR, gas, dx, dt)` pour calculer le flux.
4. Les schémas d'ordre élevé **surchargent** `compute_fluxes` pour insérer leur reconstruction avant l'appel au flux.

### Exceptions : schémas centrés

Les schémas **Lax-Friedrichs**, **Lax-Wendroff** et **JST** ne se décomposent pas en « reconstruction + solveur de Riemann ». Ce sont des schémas centrés qui calculent le flux directement à partir des valeurs de cellule, sans résoudre de problème de Riemann local. Ils ne sont pas composables avec les reconstructions d'ordre élevé.

En particulier, Lax-Wendroff inclut sa propre intégration temporelle (propriété `time_integral_included = True`), ce qui force l'utilisation de RK1 (un seul étage).

---

## §3.6 Cellules fantômes (*ghost cells*)

### Pourquoi des cellules fantômes ?

Au bord du domaine, le stencil du schéma s'étend au-delà des cellules intérieures. Par exemple, pour calculer le flux $\hat{\mathbf{F}}_{1/2}$ à l'interface gauche, un schéma d'ordre 1 a besoin de $\bar{\mathbf{U}}_0$ (la cellule à gauche du domaine), qui n'existe pas physiquement.

La solution est d'ajouter des **cellules fantômes** de chaque côté du domaine, remplies selon les conditions aux limites. Le tableau de variables conservatives passe de la forme $(3, N)$ à $(3, N + 2 \cdot n_{\text{ghost}})$.

### Nombre de cellules fantômes

Le nombre $n_{\text{ghost}}$ dépend de la largeur du stencil, donc de l'ordre du schéma :

| Reconstruction | Ordre | $n_{\text{ghost}}$ |
|---|---|---|
| Constante (ordre 1) | 1 | 1 |
| MUSCL, ENO2 | 2 | 2 |
| WENO3 | 3 | 2 |
| WENO5 | 5 | 3 |

### Types de conditions aux limites

Trois types sont implémentés dans `euler1d/boundary.py` :

#### Transmissive (zero-gradient)

Les cellules fantômes copient la cellule intérieure la plus proche :

$$\mathbf{U}_{\text{ghost}} = \mathbf{U}_{\text{bord}}$$

Physiquement, cela simule un domaine ouvert : les ondes sortent librement sans réflexion. C'est la condition par défaut pour les problèmes de Riemann (tube de Sod, test de Lax, etc.).

#### Réflective (paroi solide)

Les cellules fantômes copient la cellule intérieure symétrique avec **inversion de la quantité de mouvement** :

$$\rho_{\text{ghost}} = \rho_{\text{sym}}, \qquad (\rho u)_{\text{ghost}} = -(\rho u)_{\text{sym}}, \qquad E_{\text{ghost}} = E_{\text{sym}}$$

Physiquement, cela impose une vitesse nulle à la paroi ($u = 0$ à l'interface). Les ondes sont réfléchies à la frontière, comme sur un mur rigide.

#### Périodique

Les cellules fantômes à gauche copient les cellules intérieures de droite, et vice versa :

$$\mathbf{U}_{\text{ghost, gauche}} = \mathbf{U}_{\text{intérieur, droite}}, \qquad \mathbf{U}_{\text{ghost, droite}} = \mathbf{U}_{\text{intérieur, gauche}}$$

Physiquement, le domaine se « referme » sur lui-même : ce qui sort à droite réapparaît à gauche. C'est la condition appropriée pour les cas tests lisses (onde entropique, onde acoustique) où la solution est périodique.

---

## §3.7 Registre de schémas

### Convention de nommage

Tous les schémas sont accessibles par un nom unique suivant la convention `"reconstruction-flux"`. Par exemple :
- `"hllc"` : flux HLLC seul (reconstruction constante, ordre 1)
- `"muscl-hllc"` : reconstruction MUSCL + flux HLLC (ordre 2)
- `"weno5-roe"` : reconstruction WENO5 + flux Roe (ordre 5)

### Schémas composables et non composables

Les **flux composables** avec les reconstructions sont : Rusanov, HLL, HLLC, Roe, Godunov (5 flux). Combinés avec les 6 reconstructions (MUSCL, ENO2, WENO3, WENOZ3, WENO5, WENOZ5), ils produisent 30 schémas d'ordre élevé.

Les **schémas non composables** (Lax-Friedrichs, Lax-Wendroff, JST) ne fonctionnent qu'en ordre 1 (ou avec leur propre méthode interne). Ils sont enregistrés directement dans le registre.

Au total, le registre propose **38 schémas** : 8 autonomes + 30 combinaisons.

### Fonction `get_scheme`

La fonction `get_scheme(name, **kwargs)` du module `euler1d/schemes/__init__.py` est le point d'entrée unique pour instancier un schéma. Elle accepte le nom en minuscules et, pour les schémas MUSCL, un argument optionnel `limiter` (par défaut `"van-leer"`).

```python
from euler1d.schemes import get_scheme

# Ordre 1
scheme = get_scheme("hllc")

# Ordre 2 avec limiteur
scheme = get_scheme("muscl-hllc", limiter="superbee")

# Ordre 5
scheme = get_scheme("weno5-hllc")
```

---

## §3.8 Implémentation

Le code correspondant à ce chapitre se trouve dans trois fichiers :

- **`euler1d/boundary.py`** : conditions aux limites par cellules fantômes (§3.6). La fonction `apply_boundary_conditions` étend le tableau $(3, N)$ en $(3, N + 2 \cdot n_{\text{ghost}})$.

- **`euler1d/schemes/base.py`** : classe abstraite `NumericalScheme` (§3.5). Définit l'interface de composition reconstruction + flux, avec les propriétés `order`, `n_ghost`, `time_integral_included`, `default_time_integrator`.

- **`euler1d/schemes/__init__.py`** : registre de schémas (§3.7). La fonction `get_scheme(name)` instancie n'importe quel schéma par son nom, et `available_schemes()` retourne la liste complète.

---

## Références

- [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e éd., Springer. Chapitre 6 : « The Method of Godunov for Non-linear Systems ».
- [LeVeque, 2002] R.J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge University Press. Chapitre 4 : « Finite Volume Methods ».
