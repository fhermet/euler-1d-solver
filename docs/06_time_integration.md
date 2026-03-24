# Chapitre 6 : Intégration temporelle

> **Prérequis** : [Chapitre 3 — Méthode des volumes finis](03_finite_volume.md) (semi-discrétisation §3.3, méthode des lignes, condition CFL), [Chapitre 5 — Reconstructions d'ordre élevé](05_reconstruction.md) (ordres spatiaux 2 à 5).

## Introduction

Au chapitre 3 (§3.3), la semi-discrétisation spatiale a transformé l'EDP d'Euler en un système d'équations différentielles ordinaires (EDO) :

$$\frac{d\mathbf{U}_i}{dt} = \mathbf{L}(\mathbf{U}) \qquad \text{avec} \qquad \mathbf{L}(\mathbf{U})_i = -\frac{1}{\Delta x}\bigl(\hat{\mathbf{F}}_{i+1/2} - \hat{\mathbf{F}}_{i-1/2}\bigr)$$

L'opérateur $\mathbf{L}(\mathbf{U})$ rassemble tout le travail spatial : reconstruction des états aux interfaces (Ch. 5), calcul des flux numériques (Ch. 4) et différenciation des flux. La variable $\mathbf{U}_i(t)$ est désormais une fonction du temps seul — l'espace a été « éliminé ».

Cette approche, appelée **méthode des lignes** (*method of lines*), découple la discrétisation spatiale de la discrétisation temporelle. Il reste à choisir un **intégrateur en temps** pour faire avancer la solution de $t^n$ à $t^{n+1} = t^n + \Delta t$. Le présent chapitre détaille les cinq méthodes de Runge-Kutta implémentées dans le solveur, leur couplage avec les schémas spatiaux, et les conditions de stabilité qui gouvernent le pas de temps.

**Implémentation** : `euler1d/solver.py` — fonctions `_compute_rhs` (calcul de $\Delta t \cdot \mathbf{L}(\mathbf{U})$) et `run_simulation` (boucle en temps).

---

## §6.1 Condition CFL

### Énoncé

La condition de **Courant-Friedrichs-Lewy** (CFL) impose une borne supérieure au pas de temps pour garantir la stabilité des schémas explicites :

$$\Delta t = \text{CFL} \cdot \frac{\Delta x}{\displaystyle\max_i\bigl(|u_i| + c_i\bigr)}$$

où $u_i$ est la vitesse du fluide et $c_i = \sqrt{\gamma\, p_i / \rho_i}$ la vitesse du son dans la cellule $i$.

### Interprétation physique

Le dénominateur $\max(|u| + c)$ est la **vitesse maximale de propagation** de l'information dans le domaine. La condition CFL $\leq 1$ signifie que l'onde la plus rapide ne doit pas traverser plus d'une cellule par pas de temps. Si cette condition est violée, le domaine de dépendance numérique ne contient pas le domaine de dépendance physique, et le schéma devient instable.

### Pas de temps adaptatif

Dans le solveur, le pas de temps est **recalculé à chaque itération** :

1. Calcul de $s_{\max} = \max_i(|u_i| + c_i)$ via `max_wave_speed(U, gamma)`.
2. $\Delta t = \text{CFL} \cdot \Delta x / s_{\max}$.
3. Ajustement final : si $t + \Delta t > t_{\text{final}}$, on réduit $\Delta t$ pour atteindre exactement $t_{\text{final}}$.

Ce recalcul est essentiel car la vitesse maximale évolue au cours de la simulation (notamment lors du passage d'un choc).

---

## §6.2 Euler explicite (RK1)

### Formulation

La méthode la plus simple pour intégrer le système d'EDO est l'**Euler explicite** :

$$\mathbf{U}^{n+1} = \mathbf{U}^n + \Delta t\, \mathbf{L}(\mathbf{U}^n)$$

C'est une méthode à un seul étage, d'**ordre 1** en temps.

### Tableau de Butcher

$$\begin{array}{c|c}
0 \\
\hline
  & 1
\end{array}$$

### Utilisation

L'Euler explicite est le choix par défaut pour :

- Les **schémas spatiaux d'ordre 1** (Rusanov, HLL, HLLC, Roe, Godunov, Lax-Friedrichs) : la précision temporelle d'ordre 1 est cohérente avec l'ordre spatial.
- **Lax-Wendroff** : ce schéma centré inclut sa propre intégration temporelle via un prédicteur-correcteur de Richtmyer. L'intégrateur extérieur doit être RK1 pour éviter un double pas de temps. Le solveur force `RK1` lorsque `time_integral_included = True`.

---

## §6.3 Méthode de Heun (RK2)

### Formulation

La méthode de **Heun** (ou RK2 explicite) utilise deux étages :

$$\mathbf{k}_1 = \Delta t\, \mathbf{L}(\mathbf{U}^n)$$

$$\mathbf{k}_2 = \Delta t\, \mathbf{L}(\mathbf{U}^n + \mathbf{k}_1)$$

$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{1}{2}\bigl(\mathbf{k}_1 + \mathbf{k}_2\bigr)$$

C'est une méthode à deux étages, d'**ordre 2** en temps.

### Tableau de Butcher

$$\begin{array}{c|cc}
0 \\
1 & 1 \\
\hline
  & 1/2 & 1/2
\end{array}$$

### Utilisation

RK2 est le choix par défaut pour les reconstructions d'**ordre spatial 2** :

- **MUSCL** avec limiteur de pente (minmod, van Leer, superbee, MC)
- **ENO2** (sélection de stencil)

L'ordre temporel 2 correspond à l'ordre spatial 2, assurant un schéma globalement d'ordre 2 en espace et en temps.

---

## §6.4 SSP-RK3 (Shu-Osher)

### Motivation : la propriété SSP

Les schémas TVD (*Total Variation Diminishing*) garantissent que la variation totale de la solution ne croît pas au cours du temps, empêchant ainsi l'apparition d'oscillations parasites. Cependant, cette propriété est établie pour l'Euler explicite (RK1) sous la condition CFL $\leq 1$. Un intégrateur RK classique d'ordre supérieur ne préserve pas nécessairement la TVD.

Les méthodes **SSP** (*Strong Stability Preserving*) sont conçues pour répondre à ce problème : si l'Euler explicite préserve une propriété de stabilité (TVD, positivité, etc.) sous une certaine condition CFL, alors la méthode SSP la préserve aussi, éventuellement sous une condition CFL réduite.

### Formulation (Shu-Osher)

La méthode SSP-RK3, introduite par Shu et Osher [Shu, Osher, 1988], s'écrit sous forme de combinaisons convexes d'étapes d'Euler explicite :

$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t\, \mathbf{L}(\mathbf{U}^n)$$

$$\mathbf{U}^{(2)} = \frac{3}{4}\mathbf{U}^n + \frac{1}{4}\bigl(\mathbf{U}^{(1)} + \Delta t\, \mathbf{L}(\mathbf{U}^{(1)})\bigr)$$

$$\mathbf{U}^{n+1} = \frac{1}{3}\mathbf{U}^n + \frac{2}{3}\bigl(\mathbf{U}^{(2)} + \Delta t\, \mathbf{L}(\mathbf{U}^{(2)})\bigr)$$

C'est une méthode à trois étages, d'**ordre 3** en temps.

### Pourquoi cette écriture ?

Chaque étage est une **combinaison convexe** (coefficients positifs sommant à 1) d'un pas d'Euler explicite. Si l'Euler explicite préserve une propriété sous CFL $\leq 1$, la combinaison convexe la préserve aussi (par convexité de l'ensemble des solutions admissibles). Le coefficient SSP de cette méthode est $c = 1$ : elle préserve la TVD sous la même condition CFL que l'Euler explicite.

### Tableau de Butcher

$$\begin{array}{c|ccc}
0 \\
1 & 1 \\
1/2 & 1/4 & 1/4 \\
\hline
  & 1/6 & 1/6 & 2/3
\end{array}$$

### Utilisation

SSP-RK3 est le choix par défaut pour **WENO3** (ordres JS et Z). L'ordre temporel 3 correspond à l'ordre spatial 3, et la propriété SSP est particulièrement importante pour les reconstructions WENO qui reposent sur la préservation de la TVD.

---

## §6.5 RK4 classique

### Formulation

La méthode de Runge-Kutta d'ordre 4 classique utilise quatre étages :

$$\mathbf{k}_1 = \Delta t\, \mathbf{L}(\mathbf{U}^n)$$

$$\mathbf{k}_2 = \Delta t\, \mathbf{L}\bigl(\mathbf{U}^n + \tfrac{1}{2}\mathbf{k}_1\bigr)$$

$$\mathbf{k}_3 = \Delta t\, \mathbf{L}\bigl(\mathbf{U}^n + \tfrac{1}{2}\mathbf{k}_2\bigr)$$

$$\mathbf{k}_4 = \Delta t\, \mathbf{L}(\mathbf{U}^n + \mathbf{k}_3)$$

$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{1}{6}\bigl(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\bigr)$$

C'est une méthode à quatre étages, d'**ordre 4** en temps.

### Tableau de Butcher

$$\begin{array}{c|cccc}
0 \\
1/2 & 1/2 \\
1/2 & 0 & 1/2 \\
1 & 0 & 0 & 1 \\
\hline
  & 1/6 & 1/3 & 1/3 & 1/6
\end{array}$$

### Propriétés

- **Pas SSP** : RK4 classique n'est pas une méthode SSP. Il n'existe pas de méthode RK explicite SSP d'ordre 4 avec seulement 4 étages [Gottlieb, Shu, Tadmor, 2001]. En pratique, pour les schémas spatiaux d'ordre 4 (JST), la dissipation artificielle adaptative stabilise la solution sans nécessiter la propriété SSP.
- **Large domaine de stabilité** : la région de stabilité de RK4 dans le plan complexe est sensiblement plus grande que celles de RK1-RK3, ce qui peut autoriser des CFL légèrement plus élevés dans certaines configurations.

### Utilisation

RK4 est le choix par défaut pour le schéma **JST** (Jameson-Schmidt-Turkel), dont la dissipation artificielle adaptative requiert un intégrateur d'ordre suffisant.

---

## §6.6 Dormand-Prince (RK5)

### Motivation

Pour les reconstructions spatiales d'ordre 5 (WENO5), un intégrateur temporel d'ordre au moins 5 est nécessaire pour que l'erreur temporelle ne domine pas l'erreur spatiale. La méthode de **Dormand-Prince** [Dormand, Prince, 1980] est un schéma de Runge-Kutta à **6 étages** et d'**ordre 5**.

### Formulation

Les six étages sont définis par :

$$\mathbf{k}_1 = \Delta t\, \mathbf{L}(\mathbf{U}^n)$$

$$\mathbf{k}_2 = \Delta t\, \mathbf{L}\Bigl(\mathbf{U}^n + \tfrac{1}{5}\,\mathbf{k}_1\Bigr)$$

$$\mathbf{k}_3 = \Delta t\, \mathbf{L}\Bigl(\mathbf{U}^n + \tfrac{3}{40}\,\mathbf{k}_1 + \tfrac{9}{40}\,\mathbf{k}_2\Bigr)$$

$$\mathbf{k}_4 = \Delta t\, \mathbf{L}\Bigl(\mathbf{U}^n + \tfrac{44}{45}\,\mathbf{k}_1 - \tfrac{56}{15}\,\mathbf{k}_2 + \tfrac{32}{9}\,\mathbf{k}_3\Bigr)$$

$$\mathbf{k}_5 = \Delta t\, \mathbf{L}\Bigl(\mathbf{U}^n + \tfrac{19372}{6561}\,\mathbf{k}_1 - \tfrac{25360}{2187}\,\mathbf{k}_2 + \tfrac{64448}{6561}\,\mathbf{k}_3 - \tfrac{212}{729}\,\mathbf{k}_4\Bigr)$$

$$\mathbf{k}_6 = \Delta t\, \mathbf{L}\Bigl(\mathbf{U}^n + \tfrac{9017}{3168}\,\mathbf{k}_1 - \tfrac{355}{33}\,\mathbf{k}_2 + \tfrac{46732}{5247}\,\mathbf{k}_3 + \tfrac{49}{176}\,\mathbf{k}_4 - \tfrac{5103}{18656}\,\mathbf{k}_5\Bigr)$$

La solution avancée est :

$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{35}{384}\,\mathbf{k}_1 + \frac{500}{1113}\,\mathbf{k}_3 + \frac{125}{192}\,\mathbf{k}_4 - \frac{2187}{6784}\,\mathbf{k}_5 + \frac{11}{84}\,\mathbf{k}_6$$

On note que $\mathbf{k}_2$ n'intervient pas dans la formule finale (son coefficient $b_2$ est nul), mais il est nécessaire pour le calcul des étages suivants.

### Tableau de Butcher

$$\begin{array}{c|cccccc}
0 \\
1/5 & 1/5 \\
3/10 & 3/40 & 9/40 \\
4/5 & 44/45 & -56/15 & 32/9 \\
8/9 & 19372/6561 & -25360/2187 & 64448/6561 & -212/729 \\
1 & 9017/3168 & -355/33 & 46732/5247 & 49/176 & -5103/18656 \\
\hline
  & 35/384 & 0 & 500/1113 & 125/192 & -2187/6784 & 11/84
\end{array}$$

### Utilisation

Dormand-Prince est le choix par défaut pour **WENO5** (variantes JS et Z), assurant que l'erreur temporelle d'ordre 5 est cohérente avec la précision spatiale d'ordre 5.

> **Remarque** : la méthode de Dormand-Prince est aussi la base du solveur adaptatif `ode45` de MATLAB et `dopri5` de SciPy. Dans notre solveur, le pas de temps est contrôlé par la condition CFL et non par un estimateur d'erreur embarqué (embedded method), donc seule la formule d'ordre 5 est utilisée.

---

## §6.7 Consistance espace-temps

### Principe

L'ordre de convergence **global** d'un schéma de volumes finis est limité par le minimum de l'ordre spatial et de l'ordre temporel :

$$\text{Ordre global} = \min(\text{ordre spatial},\, \text{ordre temporel})$$

Utiliser un intégrateur RK1 (ordre 1 en temps) avec une reconstruction WENO5 (ordre 5 en espace) limiterait l'ordre global à 1 : toute la précision spatiale serait gaspillée. Inversement, utiliser RK5 avec un flux d'ordre 1 est un surcoût inutile (6 évaluations de flux par pas de temps au lieu d'une).

### Correspondance ordre spatial / intégrateur

| Ordre spatial | Reconstruction | Intégrateur par défaut | Étages |
|:---:|:---|:---|:---:|
| 1 | Constante (Rusanov, HLL, HLLC, Roe, Godunov, Lax-Friedrichs) | RK1 (Euler explicite) | 1 |
| 2 | MUSCL, ENO2 | RK2 (Heun) | 2 |
| 3 | WENO3 (JS, Z) | RK3 (SSP-RK3) | 3 |
| 4 | JST | RK4 (classique) | 4 |
| 5 | WENO5 (JS, Z) | RK5 (Dormand-Prince) | 6 |

### Logique de sélection dans le code

La propriété `default_time_integrator` de chaque schéma renvoie la méthode RK correspondant à son ordre spatial. Le solveur (`run_simulation`) applique la logique suivante :

1. Si `time_integral_included = True` (Lax-Wendroff) : forcer **RK1** pour éviter un double pas de temps.
2. Si l'utilisateur spécifie un intégrateur (`time_integrator` non nul) : utiliser celui-ci.
3. Sinon : utiliser `scheme.default_time_integrator`.

L'utilisateur peut toujours forcer un intégrateur différent (par exemple RK3 avec MUSCL pour bénéficier de la propriété SSP), mais la correspondance par défaut assure la consistance espace-temps.

---

## §6.8 Tableau récapitulatif

| Méthode | Étages | Ordre | SSP ? | Schémas spatiaux par défaut |
|:---|:---:|:---:|:---:|:---|
| RK1 (Euler explicite) | 1 | 1 | Oui (trivial) | Rusanov, HLL, HLLC, Roe, Godunov, Lax-Friedrichs, Lax-Wendroff |
| RK2 (Heun) | 2 | 2 | Non | MUSCL, ENO2 |
| RK3 (SSP-RK3, Shu-Osher) | 3 | 3 | **Oui** ($c = 1$) | WENO3 (JS, Z) |
| RK4 (classique) | 4 | 4 | Non | JST |
| RK5 (Dormand-Prince) | 6 | 5 | Non | WENO5 (JS, Z) |

**Coût par pas de temps** : chaque étage nécessite une évaluation complète de l'opérateur spatial $\mathbf{L}(\mathbf{U})$ (conditions aux limites + reconstruction + flux + différenciation). RK5 coûte donc 6 fois plus qu'Euler explicite par pas de temps, mais la précision d'ordre 5 permet d'utiliser des maillages plus grossiers pour une erreur donnée.

---

## Références

- [Shu, Osher, 1988] C.-W. Shu et S. Osher, *Efficient Implementation of Essentially Non-oscillatory Shock-Capturing Schemes*, Journal of Computational Physics, 77(2), pp. 439-471, 1988.
- [Dormand, Prince, 1980] J. R. Dormand et P. J. Prince, *A Family of Embedded Runge-Kutta Formulae*, Journal of Computational and Applied Mathematics, 6(1), pp. 19-26, 1980.
- [Toro, 2009] E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e édition, Springer, 2009, Chapitre 6.
- [Gottlieb, Shu, Tadmor, 2001] S. Gottlieb, C.-W. Shu et E. Tadmor, *Strong Stability-Preserving High-Order Time Discretization Methods*, SIAM Review, 43(1), pp. 89-112, 2001.
