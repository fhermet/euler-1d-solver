# Chapitre 1 : Équations d'Euler 1D

## Introduction

Les équations d'Euler constituent le socle de la mécanique des fluides compressibles. Elles expriment, sous forme de **lois de conservation**, trois principes fondamentaux de la physique : la conservation de la masse, de la quantité de mouvement et de l'énergie totale. En l'absence de viscosité, de conduction thermique et de forces extérieures, ces équations décrivent intégralement la dynamique d'un écoulement compressible.

L'étude du cas unidimensionnel (1D) n'est pas une simplification anecdotique. C'est le cadre minimal qui contient toute la richesse physique des écoulements compressibles : ondes de choc, détentes, discontinuités de contact. C'est pourquoi les équations d'Euler 1D servent de **problème modèle** incontournable pour :

- le développement et la validation de schémas numériques (volumes finis, différences finies),
- la compréhension de la structure des ondes dans les systèmes hyperboliques,
- l'analyse des propriétés de stabilité, dissipation et dispersion numériques.

Ce chapitre présente les équations d'Euler 1D de manière auto-suffisante. Chaque formule est accompagnée d'une explication physique terme à terme. L'implémentation numérique correspondante se trouve dans le module `euler1d/physics.py`.

---

## §1.1 Équations gouvernantes

### Forme conservative

Les équations d'Euler 1D s'écrivent sous la forme d'un **système de lois de conservation** :

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0
$$

où $\mathbf{U}$ est le **vecteur des variables conservatives** et $\mathbf{F}(\mathbf{U})$ est le **vecteur de flux physique**.

### Vecteur des variables conservatives

$$
\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ E \end{pmatrix}
$$

Chaque composante représente une **densité volumique** d'une quantité physique conservée :

| Composante | Grandeur | Signification physique |
|---|---|---|
| $\rho$ | Masse volumique | Masse de fluide par unité de volume (kg/m³). |
| $\rho u$ | Quantité de mouvement | Impulsion par unité de volume (kg/(m²·s)). C'est le produit de la densité par la vitesse $u$. |
| $E$ | Énergie totale | Énergie par unité de volume (J/m³). Elle comprend l'énergie cinétique $\frac{1}{2}\rho u^2$ et l'énergie interne $\rho e$, soit $E = \rho e + \frac{1}{2}\rho u^2$. |

### Vecteur de flux physique

$$
\mathbf{F}(\mathbf{U}) = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(E + p) \end{pmatrix}
$$

Chaque composante du flux représente le **taux de transport** de la quantité conservative correspondante à travers une section :

| Composante du flux | Signification physique |
|---|---|
| $\rho u$ | Flux de masse : débit massique par unité de surface. Le fluide transporte sa propre masse à la vitesse $u$. |
| $\rho u^2 + p$ | Flux de quantité de mouvement : il comporte deux contributions. Le terme $\rho u^2$ est le **transport convectif** de la quantité de mouvement (le fluide emporte son impulsion à la vitesse $u$). Le terme $p$ est la **force de pression** qui agit sur la section, même en l'absence d'écoulement. |
| $u(E + p)$ | Flux d'énergie : le fluide transporte son énergie totale $E$ à la vitesse $u$, plus le **travail des forces de pression** $pu$. La quantité $E + p$ est l'**enthalpie totale** par unité de volume, notée $\rho H$ avec $H = (E + p)/\rho$. |

### Interprétation physique globale

L'équation $\partial_t \mathbf{U} + \partial_x \mathbf{F} = 0$ exprime que la **variation temporelle** d'une quantité conservée dans un volume de contrôle est exactement compensée par le **flux net** entrant et sortant de ce volume. C'est la traduction mathématique du principe : *rien ne se crée, rien ne se perd, tout se transporte*.

---

## §1.2 Équation d'état

Le système de trois équations (§1.1) comporte quatre inconnues : $\rho$, $u$, $E$ et $p$. Pour fermer le système, il faut une **relation supplémentaire** liant la pression aux autres variables. C'est le rôle de l'**équation d'état**.

### Gaz parfait

Pour un gaz parfait calorifiquement parfait (hypothèse standard en dynamique des gaz), l'énergie interne spécifique $e$ est proportionnelle à la température, et la pression s'exprime :

$$
p = (\gamma - 1)\left(E - \frac{1}{2}\rho u^2\right)
$$

**Explication terme à terme :**

- $E - \frac{1}{2}\rho u^2 = \rho e$ : c'est l'**énergie interne** par unité de volume. On soustrait l'énergie cinétique $\frac{1}{2}\rho u^2$ de l'énergie totale $E$ pour obtenir la part d'énergie liée à l'agitation thermique moléculaire.
- $\gamma - 1$ : facteur de proportionnalité entre la pression et l'énergie interne. Il provient de la thermodynamique du gaz parfait.
- $p$ : la pression résultante, force par unité de surface exercée par le gaz sur ses parois.

### Le rapport des chaleurs spécifiques $\gamma$

Le paramètre $\gamma$ (gamma) est le **rapport des chaleurs spécifiques** :

$$
\gamma = \frac{c_p}{c_v}
$$

où $c_p$ est la chaleur spécifique à pression constante et $c_v$ à volume constant. Physiquement, $\gamma$ caractérise le **nombre de degrés de liberté** des molécules du gaz :

| Gaz | $\gamma$ | Degrés de liberté |
|---|---|---|
| Monoatomique (He, Ar) | $5/3 \approx 1.667$ | 3 (translation uniquement) |
| Diatomique (air, N₂, O₂) | $7/5 = 1.4$ | 5 (translation + rotation) |
| Triatomique (CO₂) | $\approx 1.3$ | 6 (translation + rotation + vibration) |

Plus $\gamma$ est élevé, plus le gaz est « rigide » : une compression produit une augmentation de pression plus importante. Pour l'air dans les conditions standard, on utilise $\gamma = 1.4$.

### Vitesse du son

La vitesse du son $a$ est la vitesse de propagation des petites perturbations de pression dans le gaz au repos :

$$
a = \sqrt{\frac{\gamma p}{\rho}}
$$

**Explication terme à terme :**

- $p/\rho$ : rapport pression/densité, qui a la dimension d'une vitesse au carré. Plus la pression est élevée (gaz comprimé) ou la densité est faible (gaz léger), plus le son se propage vite.
- $\gamma$ : le rapport des chaleurs spécifiques intervient car la propagation sonore est un processus **isentropique** (adiabatique réversible). La compressibilité isentropique diffère de la compressibilité isotherme par ce facteur $\gamma$.

La vitesse du son joue un rôle central dans les équations d'Euler car elle détermine les **vitesses de propagation des ondes** (cf. §1.4).

> **Implémentation** : les fonctions `sound_speed()` et l'équation d'état sont dans [`euler1d/physics.py`](../euler1d/physics.py).

---

## §1.3 Variables conservatives et primitives

Il existe deux représentations naturelles de l'état d'un fluide compressible. Le choix de la représentation dépend du contexte (formulation mathématique, reconstruction numérique, conditions initiales).

### Variables conservatives $\mathbf{U}$

$$
\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ E \end{pmatrix}
$$

Ces variables sont dites **conservatives** car elles apparaissent directement dans les lois de conservation : ce sont les quantités dont l'intégrale sur un volume de contrôle est conservée en l'absence de flux aux frontières. Elles sont le choix naturel pour :

- la **discrétisation en volumes finis** (la méthode repose intrinsèquement sur les bilans de quantités conservées),
- la **capture des chocs** (les relations de Rankine-Hugoniot sont formulées en variables conservatives),
- le **stockage de la solution** entre les pas de temps.

### Variables primitives $\mathbf{W}$

$$
\mathbf{W} = \begin{pmatrix} \rho \\ u \\ p \end{pmatrix}
$$

Ces variables sont dites **primitives** car elles correspondent aux grandeurs physiques directement mesurables : densité, vitesse, pression. Elles sont préférables pour :

- la **reconstruction spatiale** (MUSCL, ENO, WENO) : les variables primitives varient de manière plus régulière que les variables conservatives au voisinage des discontinuités, ce qui produit des reconstructions mieux conditionnées,
- la **spécification des conditions initiales** et aux limites (on prescrit naturellement $\rho$, $u$, $p$),
- l'**interprétation physique** des résultats.

### Conversion des variables primitives vers conservatives

Étant donnés $(\rho, u, p)$, on calcule $\mathbf{U}$ :

$$
\rho \rightarrow \rho \qquad(\text{la densité est identique dans les deux représentations})
$$

$$
\rho u = \rho \times u \qquad(\text{quantité de mouvement = densité} \times \text{vitesse})
$$

$$
E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho u^2 \qquad(\text{énergie totale = énergie interne + énergie cinétique})
$$

Le terme $p/(\gamma - 1)$ est l'énergie interne $\rho e$, obtenue en inversant l'équation d'état $p = (\gamma - 1)\rho e$.

### Conversion des variables conservatives vers primitives

Étant donné $\mathbf{U} = (\rho, \rho u, E)$, on calcule $(\rho, u, p)$ :

$$
\rho \rightarrow \rho
$$

$$
u = \frac{\rho u}{\rho} \qquad(\text{vitesse = quantité de mouvement / densité})
$$

$$
p = (\gamma - 1)\left(E - \frac{1}{2}\rho u^2\right) \qquad(\text{pression via l'équation d'état})
$$

On reconnaît l'équation d'état du §1.2. Le terme $E - \frac{1}{2}\rho u^2$ est l'énergie interne par unité de volume.

> **Implémentation** : les fonctions `primitive_to_conservative()` et `conservative_to_primitive()` sont dans [`euler1d/physics.py`](../euler1d/physics.py).

---

## §1.4 Hyperbolicité et structure propre

Le système d'Euler 1D est un système **hyperbolique** de lois de conservation. Cette propriété est fondamentale car elle garantit que l'information se propage à des vitesses finies, sous forme d'ondes bien définies.

### Forme quasi-linéaire

En dérivant le flux par rapport aux variables conservatives, le système se réécrit :

$$
\frac{\partial \mathbf{U}}{\partial t} + \mathbf{A}(\mathbf{U}) \frac{\partial \mathbf{U}}{\partial x} = 0
$$

où $\mathbf{A} = \frac{\partial \mathbf{F}}{\partial \mathbf{U}}$ est la **matrice jacobienne du flux**. Le système est hyperbolique si et seulement si $\mathbf{A}$ possède trois valeurs propres réelles et une base complète de vecteurs propres.

### Matrice jacobienne

En notant $u$ la vitesse, $a$ la vitesse du son, $H = (E + p)/\rho$ l'enthalpie totale spécifique, et $q^2 = u^2$, la jacobienne s'écrit :

$$
\mathbf{A} = \frac{\partial \mathbf{F}}{\partial \mathbf{U}} = \begin{pmatrix}
0 & 1 & 0 \\
\frac{\gamma - 3}{2}u^2 & (3 - \gamma)u & \gamma - 1 \\
\left(\frac{\gamma - 1}{2}u^2 - H\right)u & H - (\gamma - 1)u^2 & \gamma u
\end{pmatrix}
$$

**Origine des termes :** cette matrice résulte de la différentiation de chaque composante du flux $\mathbf{F}$ par rapport à chaque composante de $\mathbf{U}$. Par exemple, le terme $A_{21} = \frac{\gamma - 3}{2}u^2$ provient de $\frac{\partial(\rho u^2 + p)}{\partial \rho}$ en exprimant $p$ via l'équation d'état.

### Valeurs propres

La matrice $\mathbf{A}$ possède trois valeurs propres réelles :

$$
\lambda_1 = u - a, \qquad \lambda_2 = u, \qquad \lambda_3 = u + a
$$

**Interprétation physique :**

| Valeur propre | Vitesse de propagation | Signification |
|---|---|---|
| $\lambda_1 = u - a$ | Onde acoustique rétrograde | Perturbation de pression se propageant **vers l'amont** (dans le référentiel du fluide, à la vitesse $-a$). |
| $\lambda_2 = u$ | Onde entropique | Perturbation d'entropie (et de densité) advectée **à la vitesse du fluide**. Ne transporte aucune variation de pression ni de vitesse. |
| $\lambda_3 = u + a$ | Onde acoustique progressive | Perturbation de pression se propageant **vers l'aval** (dans le référentiel du fluide, à la vitesse $+a$). |

Les trois valeurs propres sont toujours réelles et distinctes (sauf cas dégénéré $a = 0$), ce qui confirme l'**hyperbolicité stricte** du système.

### Vecteurs propres droits

Les vecteurs propres droits $\mathbf{r}_k$ (colonnes de la matrice $\mathbf{R}$) définissent la **structure de chaque onde** — c'est-à-dire comment les variables conservatives varient à travers chaque type d'onde :

$$
\mathbf{r}_1 = \begin{pmatrix} 1 \\ u - a \\ H - ua \end{pmatrix}, \qquad
\mathbf{r}_2 = \begin{pmatrix} 1 \\ u \\ \frac{1}{2}u^2 \end{pmatrix}, \qquad
\mathbf{r}_3 = \begin{pmatrix} 1 \\ u + a \\ H + ua \end{pmatrix}
$$

**Interprétation :**

- **$\mathbf{r}_1$ et $\mathbf{r}_3$** (ondes acoustiques) : les trois composantes ($\rho$, $\rho u$, $E$) varient simultanément. Une onde acoustique modifie la densité, la vitesse **et** la pression. La troisième composante fait intervenir l'enthalpie $H$, car l'onde transporte de l'énergie.
- **$\mathbf{r}_2$** (onde entropique) : la troisième composante est $\frac{1}{2}u^2$ (énergie cinétique seule, sans contribution de pression). Cela reflète le fait que cette onde ne modifie **ni la pression ni la vitesse** — seules la densité et l'entropie varient.

### Vecteurs propres gauches

Les vecteurs propres gauches $\mathbf{l}_k$ (lignes de $\mathbf{R}^{-1}$) permettent de **projeter** un état sur les composantes de chaque onde. En notant $b_1 = \frac{\gamma - 1}{2a^2}$ et $b_2 = \frac{u}{a}$ :

$$
\mathbf{l}_1 = \frac{1}{2}\begin{pmatrix} b_1 u^2 + b_2 \\ -(b_1 \cdot 2u + 1/a) \\ 2b_1 \end{pmatrix}^T
$$

$$
\mathbf{l}_2 = \begin{pmatrix} 1 - b_1 u^2 \\ 2b_1 u \\ -2b_1 \end{pmatrix}^T
$$

$$
\mathbf{l}_3 = \frac{1}{2}\begin{pmatrix} b_1 u^2 - b_2 \\ -(b_1 \cdot 2u - 1/a) \\ 2b_1 \end{pmatrix}^T
$$

Plus explicitement, en termes de $(\gamma - 1)$, $u$ et $a$, les vecteurs propres gauches s'écrivent :

$$
\mathbf{l}_1 = \frac{1}{2a^2}\begin{pmatrix} \frac{(\gamma-1)}{2}u^2 + ua & -(\gamma-1)u - a & \gamma-1 \end{pmatrix}
$$

$$
\mathbf{l}_2 = \frac{1}{a^2}\begin{pmatrix} a^2 - \frac{(\gamma-1)}{2}u^2 & (\gamma-1)u & -(\gamma-1) \end{pmatrix}
$$

$$
\mathbf{l}_3 = \frac{1}{2a^2}\begin{pmatrix} \frac{(\gamma-1)}{2}u^2 - ua & -(\gamma-1)u + a & \gamma-1 \end{pmatrix}
$$

Ils vérifient $\mathbf{l}_i \cdot \mathbf{r}_j = \delta_{ij}$ (orthonormalité). Leur utilité principale est dans l'**analyse de Fourier** des schémas numériques (cf. `euler1d/fourier_analysis.py`) : on projette la sortie du schéma sur $\mathbf{l}_k$ pour isoler la réponse d'un mode spécifique (acoustique ou entropique).

### Décomposition caractéristique

Toute perturbation $\delta\mathbf{U}$ de la solution peut se décomposer sur la base des vecteurs propres :

$$
\delta\mathbf{U} = \sum_{k=1}^{3} \alpha_k \, \mathbf{r}_k, \qquad \alpha_k = \mathbf{l}_k \cdot \delta\mathbf{U}
$$

Chaque coefficient $\alpha_k$ est l'**amplitude de l'onde $k$**. Dans la solution, ces trois ondes se propagent indépendamment à leurs vitesses respectives $\lambda_k$. C'est cette décomposition qui est au coeur du solveur de Riemann (cf. Chapitre suivant).

---

## §1.5 Structure des ondes

Les trois familles d'ondes identifiées au §1.4 ne se comportent pas de la même manière lorsque l'amplitude des perturbations devient finie. La **nature non linéaire** des équations d'Euler crée une distinction fondamentale entre deux types de champs caractéristiques.

### Champs véritablement non linéaires (ondes acoustiques)

Les 1er et 3e champs ($\lambda_1 = u - a$ et $\lambda_3 = u + a$) sont **véritablement non linéaires** (*genuinely nonlinear*). Mathématiquement, cela signifie que :

$$
\nabla_{\mathbf{U}} \lambda_k \cdot \mathbf{r}_k \neq 0 \qquad \text{pour } k = 1, 3
$$

Autrement dit, la vitesse de propagation $\lambda_k$ **varie le long de l'onde**. Physiquement, cela a une conséquence majeure : une onde acoustique de compression a tendance à se **raidir** (les crêtes rattrapent les creux), tandis qu'une onde de détente a tendance à s'**étaler**.

Concrètement, chaque onde acoustique peut se manifester sous deux formes exclusives :

#### Onde de choc (compression)

Un choc est une **discontinuité** dans la solution. Il se forme lorsqu'une onde de compression non linéaire se raidit jusqu'à devenir verticale. À travers un choc :

- la densité $\rho$, la pression $p$ et la vitesse $u$ subissent des sauts finis,
- l'entropie **augmente** (processus irréversible),
- le choc se propage à une vitesse $s$ déterminée par les **relations de Rankine-Hugoniot** :

$$
s[\![\mathbf{U}]\!] = [\![\mathbf{F}(\mathbf{U})]\!]
$$

où $[\![Q]\!] = Q_R - Q_L$ désigne le saut de la quantité $Q$ à travers la discontinuité.

#### Onde de détente (raréfaction)

Une détente est une onde **continue et lisse** (fan de raréfaction). Elle apparaît lorsque le fluide se détend (diminution de pression). Dans une détente :

- toutes les variables varient continûment,
- l'entropie est **constante** (processus isentropique),
- la solution est **auto-similaire** : elle ne dépend que du rapport $x/t$.

Les invariants de Riemann à travers une détente sont :

- 1er champ : $u + \frac{2a}{\gamma - 1} = \text{const}$ (invariant de Riemann le long de $\lambda_1$),
- 3e champ : $u - \frac{2a}{\gamma - 1} = \text{const}$ (invariant de Riemann le long de $\lambda_3$).

### Champ linéairement dégénéré (onde entropique)

Le 2e champ ($\lambda_2 = u$) est **linéairement dégénéré** (*linearly degenerate*). Mathématiquement :

$$
\nabla_{\mathbf{U}} \lambda_2 \cdot \mathbf{r}_2 = 0
$$

La vitesse de propagation $\lambda_2 = u$ est **constante le long de l'onde**. L'onde ne se raidit ni ne s'étale : elle conserve sa forme. Elle se manifeste sous une forme unique :

#### Discontinuité de contact

Une discontinuité de contact est une **surface matérielle** se déplaçant à la vitesse du fluide $u$. À travers cette discontinuité :

- la **densité** $\rho$ et la **température** $T$ subissent un saut,
- la **vitesse** $u$ et la **pression** $p$ sont **continues**,
- l'entropie subit un saut, mais il n'y a **pas de production d'entropie** (contrairement au choc),
- il n'y a **pas de flux de masse** à travers la discontinuité dans le référentiel de celle-ci.

Physiquement, une discontinuité de contact sépare deux masses de fluide ayant des densités (ou températures) différentes mais la même pression et la même vitesse. C'est l'analogue d'une interface entre deux fluides qui glissent ensemble sans se mélanger.

### Résumé : structure d'un problème de Riemann

Le problème de Riemann est la résolution des équations d'Euler avec des **conditions initiales constantes par morceaux**, séparées par une discontinuité en $x = x_0$. La solution est composée de trois ondes séparées par quatre états constants :

```
    Onde 1          Onde 2          Onde 3
  (acoustique)   (entropique)    (acoustique)

 UL | détente  | U*L | contact | U*R | choc   | UR
    | ou choc  |     |         |     | ou     |
    |          |     |         |     | détente|
```

- **UL** et **UR** : états gauche et droit (données initiales).
- **U\*L** et **U\*R** : états intermédiaires. Ils partagent la **même pression** $p^\ast$ et la **même vitesse** $u^\ast$ (conditions de compatibilité à travers la discontinuité de contact). Ils diffèrent uniquement par leur densité.
- **Onde 1** ($\lambda_1$) : choc ou détente séparant UL de U\*L.
- **Onde 2** ($\lambda_2$) : discontinuité de contact séparant U\*L de U\*R.
- **Onde 3** ($\lambda_3$) : choc ou détente séparant U\*R de UR.

La résolution du problème de Riemann consiste à trouver $(p^\ast, u^\ast)$ puis à déterminer la nature (choc ou détente) de chaque onde acoustique. Ce problème est au coeur des schémas de type Godunov et de tous les solveurs de Riemann approchés (HLL, HLLC, Roe) présentés dans les chapitres suivants.

> **Implémentation** : le solveur de Riemann exact est dans `euler1d/riemann.py`. Les cas test (Sod, Lax, double détente) sont dans `euler1d/test_cases.py`.

---

## §1.6 Implémentation dans le code

Les concepts mathématiques de ce chapitre sont implémentés dans les modules suivants :

| Concept | Module | Fonctions |
|---|---|---|
| Équation d'état (§1.2) | `euler1d/physics.py` | `sound_speed()` |
| Conversion primitive → conservative (§1.3) | `euler1d/physics.py` | `primitive_to_conservative()` |
| Conversion conservative → primitive (§1.3) | `euler1d/physics.py` | `conservative_to_primitive()` |
| Flux physique (§1.1) | `euler1d/physics.py` | `compute_flux()` |
| Vitesse maximale d'onde (§1.4) | `euler1d/physics.py` | `max_wave_speed()` |
| Propriétés du gaz (§1.2) | `euler1d/config.py` | `GasProperties` |
| Problème de Riemann (§1.5) | `euler1d/riemann.py` | Solveur exact |

Les variables conservatives sont stockées dans des tableaux NumPy de forme `(3, N)` où `N` est le nombre de cellules :
- ligne 0 : $\rho$
- ligne 1 : $\rho u$
- ligne 2 : $E$

Cette convention est utilisée de manière cohérente dans tout le code.

---

## Références

- **[Toro, 2009]** E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e édition, Springer, 2009. Chapitres 1 à 3 : formulation des équations d'Euler, équation d'état, structure des ondes, problème de Riemann.

- **[LeVeque, 2002]** R.J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge University Press, 2002. Chapitre 1 : lois de conservation, hyperbolicité, ondes caractéristiques.

- **[Laney, 1998]** C.B. Laney, *Computational Gasdynamics*, Cambridge University Press, 1998. Chapitres 2-5 : dérivation physique des équations d'Euler, thermodynamique du gaz parfait.
