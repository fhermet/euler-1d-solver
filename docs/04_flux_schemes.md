# Chapitre 4 : Flux numériques d'ordre 1

> **Prérequis** : [Chapitre 1 — Équations d'Euler](01_euler_equations.md) (variables conservatives et primitives, flux physique), [Chapitre 2 — Problème de Riemann](02_riemann_problem.md) (structure en trois ondes, solution exacte), [Chapitre 3 — Méthode des volumes finis](03_finite_volume.md) (flux numérique, semi-discrétisation, composition reconstruction + flux).

## Introduction

Le chapitre 3 a montré que le cœur de la méthode des volumes finis est le **flux numérique** $\hat{\mathbf{F}}_{i+1/2} = \hat{\mathbf{F}}(\mathbf{U}_L, \mathbf{U}_R)$ à chaque interface. Ce chapitre détaille les huit schémas de flux implémentés dans le code, tous autonomes (reconstruction constante par morceaux, donc d'ordre 1 en espace — sauf Lax-Wendroff et JST qui sont intrinsèquement d'ordre 2).

### Taxonomie des flux numériques

On distingue deux familles de flux numériques :

**Schémas décentrés (*upwind*)** — Ils exploitent la structure d'ondes du problème de Riemann pour propager l'information dans le sens physique. Chaque schéma résout (exactement ou approximativement) le problème de Riemann local à l'interface :

- **Godunov** (§4.1) : solution exacte du problème de Riemann.
- **Rusanov** (§4.2) : une seule vitesse d'onde maximale.
- **HLL** (§4.3) : deux ondes (gauche et droite), un état intermédiaire.
- **HLLC** (§4.4) : trois ondes, incluant l'onde de contact.
- **Roe** (§4.5) : linéarisation exacte de la matrice de flux.

Ces cinq schémas sont **composables** avec les reconstructions d'ordre élevé (MUSCL, ENO, WENO) : on peut remplacer la reconstruction constante par une reconstruction d'ordre supérieur sans modifier le solveur de Riemann (cf. §3.5, Chapitre 5).

**Schémas centrés** — Ils n'utilisent pas de solveur de Riemann. Le flux est une moyenne des flux physiques gauche et droit, corrigée par un terme de dissipation numérique :

- **Lax-Friedrichs** (§4.6) : dissipation proportionnelle à $\Delta x / \Delta t$ (globale, non locale).
- **Lax-Wendroff** (§4.7) : prédicteur-correcteur, ordre 2 en espace et en temps.
- **JST** (§4.8) : dissipation artificielle adaptative (ordre 2 effectif avec capteur de choc).

Ces trois schémas **ne sont pas composables** avec les reconstructions d'ordre élevé : leur formulation ne se décompose pas en « reconstruction + solveur de Riemann ».

### Notations communes

Dans tout le chapitre, pour une interface $i+1/2$ entre la cellule $i$ (état gauche $\mathbf{U}_L$) et la cellule $i+1$ (état droit $\mathbf{U}_R$) :

- $\rho_K$, $u_K$, $p_K$ : variables primitives de l'état $K \in \{L, R\}$.
- $a_K = \sqrt{\gamma p_K / \rho_K}$ : vitesse du son.
- $\mathbf{F}_K = \mathbf{F}(\mathbf{U}_K)$ : flux physique évalué à partir de l'état $K$.
- $E_K = p_K / (\gamma - 1) + \tfrac{1}{2} \rho_K u_K^2$ : énergie totale.
- $H_K = (E_K + p_K) / \rho_K$ : enthalpie totale spécifique.

---

## §4.1 Schéma de Godunov

### Motivation

Le schéma de Godunov est le schéma de référence historique. Il repose sur l'idée la plus naturelle : puisque chaque interface entre deux cellules constitue un problème de Riemann, **résolvons-le exactement**. Ce schéma est optimal au sens où il n'introduit aucune approximation dans l'évaluation du flux — seule la reconstruction constante par morceaux limite la précision à l'ordre 1.

### Principe

À chaque interface $i+1/2$, on résout le problème de Riemann exact entre l'état $\mathbf{U}_L = \mathbf{U}_i$ et l'état $\mathbf{U}_R = \mathbf{U}_{i+1}$ (cf. Chapitre 2). La solution de ce problème de Riemann est auto-similaire : elle ne dépend que de $\xi = x/t$. Le flux numérique est obtenu en évaluant le flux physique à l'interface, c'est-à-dire en $\xi = 0$ :

$$\hat{\mathbf{F}}_{i+1/2} = \mathbf{F}\bigl(\mathbf{U}^{\text{Riemann}}(0;\, \mathbf{U}_L, \mathbf{U}_R)\bigr)$$

L'évaluation de $\mathbf{U}^{\text{Riemann}}(0)$ nécessite de déterminer dans quelle région de la solution (gauche, étoile gauche, étoile droite, ou droite) se situe l'interface $\xi = 0$, puis de calculer les variables primitives correspondantes.

### Formule

Le flux est simplement le flux physique évalué à l'état solution en $\xi = 0$ :

$$\hat{\mathbf{F}}_{i+1/2} = \begin{pmatrix} \rho^* u^* \\ \rho^* (u^*)^2 + p^* \\ u^* (E^* + p^*) \end{pmatrix}$$

où $(\rho^*, u^*, p^*)$ sont les variables primitives de la solution du problème de Riemann en $\xi = 0$, et $E^* = p^* / (\gamma - 1) + \tfrac{1}{2} \rho^* (u^*)^2$ l'énergie totale correspondante.

La détermination de $(\rho^*, u^*, p^*)$ passe par :
1. Le calcul de la pression intermédiaire $p^*$ par itération de Newton (cf. §2.4).
2. L'identification du type de chaque onde (choc ou détente) à partir du signe de $p^* - p_L$ et $p^* - p_R$.
3. L'échantillonnage de la solution en $\xi = 0$ selon la position relative des ondes par rapport à l'interface (cf. §2.5).

### Avantages

- **Exactitude maximale** : aucune approximation dans l'évaluation du flux (à reconstruction donnée). Toutes les ondes (chocs, détentes, contacts) sont résolues avec une précision optimale.
- **Schéma de référence** : sert de point de comparaison pour tous les autres solveurs approchés.
- **Robustesse** : fonctionne pour tous les cas, y compris les grandes différences de pression.

### Inconvénients

- **Coût élevé** : chaque interface nécessite une itération de Newton (typiquement 3 à 6 itérations) pour trouver $p^*$, plus l'échantillonnage de la solution. Cela représente un coût bien supérieur aux solveurs approchés.
- **Boucle Python** : dans l'implémentation, le solveur exact est appelé interface par interface dans une boucle Python (car `sample_at_interface` ne se vectorise pas trivialement), ce qui le rend significativement plus lent que les schémas vectorisés (Rusanov, HLL).
- **Reste d'ordre 1** : malgré l'exactitude du flux, la reconstruction constante par morceaux limite la précision globale à l'ordre 1.

### Comparaison

Le schéma de Godunov est la référence en termes de qualité de flux, mais son coût le rend peu compétitif en pratique. Le schéma HLLC (§4.4) offre une qualité de flux comparable (résolution des trois ondes) pour un coût bien moindre. Le schéma de Roe (§4.5) est aussi une bonne approximation, mais peut nécessiter une correction entropique.

### Références

- [Godunov, 1959] S.K. Godunov, « A difference method for numerical calculation of discontinuous solutions of the equations of hydrodynamics », *Mat. Sb.*, 47(3), pp. 271-306.
- [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e éd., Springer, §6.2.

---

## §4.2 Schéma de Rusanov (Lax-Friedrichs local)

### Motivation

Le schéma de Godunov (§4.1) est coûteux car il résout le problème de Riemann exact à chaque interface. La question est : peut-on obtenir un schéma stable et conservatif avec une estimation beaucoup plus simple du flux ? Le schéma de Rusanov répond par l'affirmative en utilisant une seule vitesse d'onde maximale pour borner la propagation de l'information.

### Principe

Le schéma de Rusanov est le plus simple des solveurs de Riemann approchés de type décentré. Il remplace la structure complexe en trois ondes du problème de Riemann par un **modèle à une seule onde** : toute l'information se propage à la vitesse maximale $S_{\max}$. Le flux est la moyenne des flux physiques, corrigée par un terme de dissipation numérique proportionnel au saut de variables conservatives.

### Formule

$$\hat{\mathbf{F}}_{i+1/2} = \frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \frac{1}{2} S_{\max} (\mathbf{U}_R - \mathbf{U}_L)$$

avec la vitesse d'onde maximale locale :

$$S_{\max} = \max(|u_L| + a_L,\; |u_R| + a_R)$$

**Interprétation terme par terme** :

- $\frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R)$ : **flux centré**, moyenne arithmétique des flux physiques de part et d'autre de l'interface. En soi, ce terme est instable car il ne tient pas compte du sens de propagation.

- $-\frac{1}{2} S_{\max} (\mathbf{U}_R - \mathbf{U}_L)$ : **terme de dissipation numérique**. Il est proportionnel au saut des variables conservatives et à la vitesse maximale locale. Ce terme stabilise le schéma en ajoutant une diffusion artificielle. Il est analogue à un terme de viscosité artificielle $\nu \cdot \partial^2 \mathbf{U} / \partial x^2$ avec $\nu \propto S_{\max} \cdot \Delta x$.

La vitesse $S_{\max}$ borne les trois vitesses propres du système ($u - a$, $u$, $u + a$) : on a toujours $|u - a| \leq |u| + a$ et $|u + a| \leq |u| + a$. Utiliser cette borne unique pour les trois ondes est ce qui rend le schéma simple mais excessivement dissipatif.

### Avantages

- **Simplicité** : la formule ne nécessite que les flux physiques et les vitesses du son. Pas d'itération, pas de décomposition en ondes.
- **Robustesse** : la dissipation importante garantit la stabilité même dans des cas difficiles (forts gradients, rapport de pression élevé).
- **Vectorisation complète** : toutes les opérations sont des opérations numpy élément par élément, sans boucle Python.
- **Composable** : peut être combiné avec MUSCL, ENO ou WENO pour monter en ordre.

### Inconvénients

- **Très dissipatif** : le schéma le plus dissipatif parmi les solveurs de Riemann approchés. Une seule vitesse d'onde pour les trois familles signifie que l'onde de contact (qui se propage à la vitesse $u$) est amortie à un taux proportionnel à $|u| + a$ au lieu de $|u|$. Les discontinuités de contact sont fortement étalées.
- **Précision limitée** : sur les cas test de type tube à choc, les profils sont nettement plus smearés que ceux de HLL (§4.3), HLLC (§4.4) ou Roe (§4.5).

### Comparaison

Rusanov est à Godunov (§4.1) ce que l'estimation grossière est à la solution exacte : on sacrifie toute la structure des ondes pour gagner en simplicité et en coût. HLL (§4.3) améliore Rusanov en distinguant deux vitesses d'onde (gauche et droite), ce qui réduit la dissipation. HLLC (§4.4) va plus loin en ajoutant l'onde de contact. Le schéma de Lax-Friedrichs global (§4.6) est encore plus dissipatif que Rusanov car il utilise $\Delta x / \Delta t$ au lieu de la vitesse locale.

### Références

- [Rusanov, 1961] V.V. Rusanov, « The calculation of the interaction of non-stationary shock waves with barriers », *Zh. Vychisl. Mat. i Mat. Fiz.*, 1(2), pp. 267-279.
- [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e éd., Springer, §10.5.

---

## §4.3 Schéma HLL

### Motivation

Le schéma de Rusanov (§4.2) utilise une seule vitesse d'onde pour les trois familles, ce qui surestime la dissipation. Le schéma HLL (Harten, Lax, van Leer) améliore cette approche en utilisant **deux vitesses d'onde** — une pour l'onde la plus rapide à gauche ($S_L$) et une pour l'onde la plus rapide à droite ($S_R$) — et en calculant un **état intermédiaire unique** entre ces deux ondes.

### Principe

Le problème de Riemann exact produit trois ondes et quatre états. Le modèle HLL simplifie cette structure en ne retenant que **deux ondes** (les plus rapides) et **trois états** :

$$\mathbf{U}_L \xrightarrow{S_L} \mathbf{U}^*_{\text{HLL}} \xrightarrow{S_R} \mathbf{U}_R$$

L'état intermédiaire $\mathbf{U}^*_{\text{HLL}}$ est déterminé par la conservation intégrale : en intégrant les équations de conservation sur le domaine de dépendance borné par $S_L$ et $S_R$, on obtient :

$$\mathbf{U}^*_{\text{HLL}} = \frac{S_R \mathbf{U}_R - S_L \mathbf{U}_L + \mathbf{F}_L - \mathbf{F}_R}{S_R - S_L}$$

Le flux à l'interface dépend de la position de l'interface par rapport aux deux ondes.

### Formule

Le flux HLL s'écrit selon trois cas :

$$\hat{\mathbf{F}}_{i+1/2} = \begin{cases} \mathbf{F}_L & \text{si } S_L \geq 0 \\ \mathbf{F}_{\text{HLL}} & \text{si } S_L < 0 < S_R \\ \mathbf{F}_R & \text{si } S_R \leq 0 \end{cases}$$

avec le flux intermédiaire :

$$\mathbf{F}_{\text{HLL}} = \frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R (\mathbf{U}_R - \mathbf{U}_L)}{S_R - S_L}$$

**Interprétation terme par terme du flux intermédiaire** :

- $S_R \mathbf{F}_L - S_L \mathbf{F}_R$ : moyenne pondérée des flux physiques par les vitesses d'onde. Comme $S_L < 0 < S_R$, les deux termes contribuent positivement. Le poids de chaque flux est proportionnel à la vitesse de l'onde opposée.

- $S_L S_R (\mathbf{U}_R - \mathbf{U}_L)$ : terme de dissipation numérique. Comme $S_L < 0$ et $S_R > 0$, le produit $S_L S_R$ est négatif. Ce terme ajoute de la dissipation proportionnelle au saut des variables conservatives, pondérée par le produit des vitesses d'onde.

- $(S_R - S_L)^{-1}$ : facteur de normalisation qui assure la consistance ($\hat{\mathbf{F}}(\mathbf{U}, \mathbf{U}) = \mathbf{F}(\mathbf{U})$).

### Estimation des vitesses d'onde (Davis)

Les vitesses d'onde $S_L$ et $S_R$ doivent borner les ondes les plus rapides de chaque côté. L'estimation de Davis est simple et robuste :

$$S_L = \min(u_L - a_L,\; u_R - a_R)$$
$$S_R = \max(u_L + a_L,\; u_R + a_R)$$

Cette estimation prend le minimum (resp. maximum) des vitesses caractéristiques gauche et droite, ce qui garantit que $S_L$ et $S_R$ encadrent bien toutes les ondes réelles.

### Avantages

- **Moins dissipatif que Rusanov** : deux vitesses d'onde au lieu d'une permettent un meilleur encadrement de la zone de propagation.
- **Vectorisé** : implémenté avec des masques numpy, sans boucle Python. Rapide.
- **Robuste** : les estimations de Davis sont toujours valides et le schéma est positivement conservatif (préserve la positivité de la densité et de la pression sous condition CFL).
- **Composable** : combinable avec les reconstructions d'ordre élevé.

### Inconvénients

- **Étalement des contacts** : avec un seul état intermédiaire, HLL ne peut pas résoudre la discontinuité de contact. Dans la solution exacte, $u$ et $p$ sont continus à travers le contact mais $\rho$ est discontinu. HLL « moyenne » cette discontinuité dans son état unique $\mathbf{U}^*_{\text{HLL}}$, ce qui étale le profil de densité.
- **Moins précis que HLLC ou Roe** sur les cas avec des contacts marqués (tube de Sod, test de Lax).

### Comparaison

HLL améliore Rusanov (§4.2) en distinguant deux vitesses d'onde. Mais il ne résout que deux ondes sur trois : la discontinuité de contact est manquante. HLLC (§4.4) corrige ce défaut en ajoutant une troisième onde (l'onde de contact $S^*$). Roe (§4.5) offre une résolution complète des trois ondes par linéarisation, mais peut souffrir de chocs d'expansion sans correction entropique. Godunov (§4.1) est exact mais plus coûteux.

### Références

- [Harten, Lax, van Leer, 1983] A. Harten, P.D. Lax, B. van Leer, « On upstream differencing and Godunov-type schemes for hyperbolic conservation laws », *SIAM Review*, 25(1), pp. 35-61.
- [Davis, 1988] S.F. Davis, « Simplified second-order Godunov-type methods », *SIAM J. Sci. Stat. Comput.*, 9(3), pp. 445-473.

---

## §4.4 Schéma HLLC

> **Section la plus détaillée de ce chapitre** — Le schéma HLLC est le solveur de Riemann approché le plus utilisé en pratique. Il combine un excellent rapport qualité/coût avec une résolution complète des trois ondes du système d'Euler.

### Motivation

Le schéma HLL (§4.3) modélise le problème de Riemann avec seulement deux ondes ($S_L$ et $S_R$) et un seul état intermédiaire. Cette simplification a un prix : l'**onde de contact** — la deuxième onde du système d'Euler, associée à la valeur propre $\lambda_2 = u$ — est absente du modèle. Physiquement, l'onde de contact sépare deux régions à la même pression et la même vitesse, mais avec des densités différentes (cf. §2.1). En l'absence de cette onde, HLL « moyenne » les deux densités, ce qui étale le profil de densité au voisinage du contact.

Le schéma HLLC (*HLL-Contact*) corrige ce défaut en ajoutant une **troisième onde** $S^*$ correspondant à l'onde de contact. Le problème de Riemann approché comporte alors trois ondes et quatre états, comme le problème exact.

### Principe : quatre régions

Le modèle HLLC divise le plan $(x, t)$ en quatre régions séparées par trois ondes :

$$\mathbf{U}_L \xrightarrow{S_L} \mathbf{U}^*_L \xrightarrow{S^*} \mathbf{U}^*_R \xrightarrow{S_R} \mathbf{U}_R$$

```
    t
    ^
    |    S_L       S*       S_R
    |   /          |          \
    |  /   U*_L    |   U*_R    \
    | /            |            \
    |/     U_L     |     U_R     \
    +------------------------------------> x
```

- **Région gauche** ($\xi < S_L$) : état non perturbé $\mathbf{U}_L$.
- **Région étoile gauche** ($S_L < \xi < S^*$) : état intermédiaire $\mathbf{U}^*_L$.
- **Région étoile droite** ($S^* < \xi < S_R$) : état intermédiaire $\mathbf{U}^*_R$.
- **Région droite** ($\xi > S_R$) : état non perturbé $\mathbf{U}_R$.

Les deux états étoile partagent la **même pression** $p^*$ et la **même vitesse** $u^* = S^*$ (conditions de Rankine-Hugoniot à travers l'onde de contact), mais ont des **densités différentes** $\rho^*_L \neq \rho^*_R$. C'est précisément cette différence de densité que HLL ne pouvait pas capturer.

### Étape 1 : Estimation des vitesses d'onde $S_L$ et $S_R$

On utilise les estimations de Davis, identiques à celles de HLL (§4.3) :

$$S_L = \min(u_L - a_L,\; u_R - a_R)$$
$$S_R = \max(u_L + a_L,\; u_R + a_R)$$

Ces estimations garantissent que $S_L$ et $S_R$ encadrent les ondes acoustiques réelles du problème de Riemann exact.

### Étape 2 : Vitesse de contact $S^*$

La vitesse de l'onde de contact $S^*$ est déterminée par la conservation de la quantité de mouvement à travers les ondes. En écrivant les relations de Rankine-Hugoniot pour les deux ondes extérieures (entre $\mathbf{U}_L$ et $\mathbf{U}^*_L$ d'une part, $\mathbf{U}^*_R$ et $\mathbf{U}_R$ d'autre part), et en imposant l'égalité des pressions étoile ($p^*_L = p^*_R$), on obtient :

$$S^* = \frac{p_R - p_L + \rho_L u_L (S_L - u_L) - \rho_R u_R (S_R - u_R)}{\rho_L (S_L - u_L) - \rho_R (S_R - u_R)}$$

**Interprétation physique** :

- **Numérateur** : $p_R - p_L$ est le saut de pression à travers l'interface. Les termes $\rho_K u_K (S_K - u_K)$ représentent le flux de quantité de mouvement à travers chaque onde extérieure, vu dans le référentiel de l'onde.

- **Dénominateur** : $\rho_K (S_K - u_K)$ est le débit massique à travers chaque onde extérieure. La différence des deux débits normalise l'expression.

- **Résultat** : $S^*$ est la vitesse commune des états étoile. Elle correspond à la vitesse du fluide dans la zone intermédiaire, c'est-à-dire la vitesse de la discontinuité de contact.

### Étape 3 : États intermédiaires $\mathbf{U}^*_K$

Les états intermédiaires sont obtenus par les relations de Rankine-Hugoniot à travers les ondes extérieures. Pour un côté $K \in \{L, R\}$ avec la vitesse d'onde $S_K$ :

$$\mathbf{U}^*_K = \rho_K \frac{S_K - u_K}{S_K - S^*} \begin{pmatrix} 1 \\ S^* \\ E_K / \rho_K + (S^* - u_K)\bigl[S^* + p_K / (\rho_K (S_K - u_K))\bigr] \end{pmatrix}$$

**Interprétation terme par terme** :

- **Facteur $\rho_K (S_K - u_K) / (S_K - S^*)$** : ce ratio traduit la compression ou la dilatation du fluide entre l'état $K$ et l'état étoile. Il est déterminé par la conservation de la masse à travers l'onde $S_K$.

- **Composante 1** (masse) : la densité étoile est $\rho^*_K = \rho_K (S_K - u_K) / (S_K - S^*)$.

- **Composante 2** (quantité de mouvement) : la vitesse dans la zone étoile est $S^*$, donc $(\rho u)^*_K = \rho^*_K \cdot S^*$.

- **Composante 3** (énergie) : l'énergie totale spécifique est modifiée par le travail des forces de pression lors du passage à travers l'onde. Le terme $p_K / (\rho_K (S_K - u_K))$ représente le rapport entre la pression et le débit massique à travers l'onde.

### Étape 4 : Flux HLLC

Le flux à l'interface est déterminé par la position de l'interface ($\xi = 0$) par rapport aux trois ondes :

$$\hat{\mathbf{F}}_{i+1/2} = \begin{cases} \mathbf{F}_L & \text{si } S_L \geq 0 \\[4pt] \mathbf{F}^*_L = \mathbf{F}_L + S_L (\mathbf{U}^*_L - \mathbf{U}_L) & \text{si } S_L < 0 \leq S^* \\[4pt] \mathbf{F}^*_R = \mathbf{F}_R + S_R (\mathbf{U}^*_R - \mathbf{U}_R) & \text{si } S^* < 0 < S_R \\[4pt] \mathbf{F}_R & \text{si } S_R \leq 0 \end{cases}$$

**Interprétation des quatre cas** :

1. **$S_L \geq 0$** : toutes les ondes se propagent vers la droite. L'interface « voit » l'état gauche non perturbé. Le flux est simplement $\mathbf{F}_L$.

2. **$S_L < 0 \leq S^*$** : l'onde la plus rapide à gauche a déjà traversé l'interface, mais l'onde de contact est encore à droite. L'interface est dans la région étoile gauche. Le flux $\mathbf{F}^*_L$ est obtenu par la relation de Rankine-Hugoniot : $\mathbf{F}^*_L = \mathbf{F}_L + S_L (\mathbf{U}^*_L - \mathbf{U}_L)$.

3. **$S^* < 0 < S_R$** : l'onde de contact a traversé l'interface, mais l'onde droite ne l'a pas encore atteinte. L'interface est dans la région étoile droite. Le flux $\mathbf{F}^*_R$ est obtenu de manière symétrique.

4. **$S_R \leq 0$** : toutes les ondes se propagent vers la gauche. L'interface « voit » l'état droit non perturbé.

### Avantages

- **Résolution des trois ondes** : contrairement à HLL, HLLC capture correctement la discontinuité de contact. Le profil de densité est nettement plus net.
- **Coût modéré** : pas d'itération de Newton (contrairement à Godunov). Les formules sont algébriques explicites.
- **Robustesse** : les estimations de Davis garantissent la positivité de la densité et de la pression sous condition CFL. HLLC est plus robuste que Roe face aux cas difficiles (double détente, faibles densités).
- **Composable** : combinable avec MUSCL, ENO, WENO pour monter en ordre.

### Inconvénients

- **Boucle Python** : dans l'implémentation actuelle, le flux est calculé interface par interface dans une boucle Python (à cause des quatre cas conditionnels). Cela le rend plus lent que HLL (vectorisé) pour un grand nombre de cellules, bien que la qualité du flux soit supérieure.
- **Approximation des vitesses d'onde** : la qualité du résultat dépend de l'estimation de $S_L$ et $S_R$. Les estimations de Davis sont simples mais peuvent être sous-optimales dans certains cas.

### Comparaison

| Aspect | Godunov (§4.1) | Rusanov (§4.2) | HLL (§4.3) | **HLLC** | Roe (§4.5) |
|--------|----------------|----------------|------------|----------|------------|
| Ondes résolues | 3 (exact) | 1 | 2 | **3** | 3 |
| Contacts | Exact | Très étalé | Étalé | **Net** | Net |
| Itération Newton | Oui | Non | Non | **Non** | Non |
| Robustesse | Excellente | Excellente | Bonne | **Bonne** | Correction requise |
| Coût relatif | Élevé | Faible | Faible | **Moyen** | Moyen |

HLLC offre le meilleur compromis entre qualité de flux et coût de calcul. C'est pourquoi il est le solveur de Riemann approché le plus recommandé pour les équations d'Euler, et le flux par défaut utilisé dans les combinaisons d'ordre élevé (MUSCL-HLLC, WENO3-HLLC, WENO5-HLLC).

### Références

- [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e éd., Springer, §10.4.
- [Batten et al., 1997] P. Batten, N. Clarke, C. Lambert, D.M. Causon, « On the choice of wavespeeds for the HLLC Riemann solver », *SIAM J. Sci. Comput.*, 18(6), pp. 1553-1570.

---

## §4.5 Schéma de Roe

### Motivation

Les schémas de type HLL/HLLC (§4.3, §4.4) approchent le problème de Riemann en simplifiant la structure des ondes. Le schéma de Roe adopte une approche différente : il **linéarise** le système non linéaire autour d'un état moyen judicieusement choisi, puis résout *exactement* le problème de Riemann *linéarisé*. L'idée est de capturer les trois ondes avec leurs amplitudes correctes, comme Godunov (§4.1), mais sans itération de Newton.

### Principe

Le système d'Euler non linéaire $\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) = 0$ est remplacé localement par un système linéaire :

$$\frac{\partial \mathbf{U}}{\partial t} + \tilde{\mathbf{A}} \frac{\partial \mathbf{U}}{\partial x} = 0$$

où $\tilde{\mathbf{A}} = \tilde{\mathbf{A}}(\mathbf{U}_L, \mathbf{U}_R)$ est la **matrice de Roe**, une matrice constante qui satisfait les trois conditions de Roe :

1. **Hyperbolique** : $\tilde{\mathbf{A}}$ est diagonalisable à valeurs propres réelles.
2. **Consistante** : $\tilde{\mathbf{A}}(\mathbf{U}, \mathbf{U}) = \mathbf{A}(\mathbf{U}) = \mathbf{F}'(\mathbf{U})$, la jacobienne exacte.
3. **Conservation** : $\tilde{\mathbf{A}} (\mathbf{U}_R - \mathbf{U}_L) = \mathbf{F}_R - \mathbf{F}_L$ (relation de Rankine-Hugoniot discrète).

La condition 3 est cruciale : elle garantit qu'un choc isolé est capturé exactement (en un seul point) par le schéma.

### Moyennes de Roe

Pour le système d'Euler, Roe a montré que la matrice satisfaisant ces trois conditions est obtenue en évaluant la jacobienne usuelle $\mathbf{A}(\mathbf{U})$ aux **moyennes de Roe** suivantes :

$$\tilde{u} = \frac{\sqrt{\rho_L}\, u_L + \sqrt{\rho_R}\, u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$

$$\tilde{H} = \frac{\sqrt{\rho_L}\, H_L + \sqrt{\rho_R}\, H_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$$

$$\tilde{a} = \sqrt{(\gamma - 1)\left(\tilde{H} - \tfrac{1}{2}\tilde{u}^2\right)}$$

Ces moyennes pondérées par $\sqrt{\rho}$ ne sont pas des moyennes arithmétiques : elles donnent plus de poids au côté le plus dense. C'est cette pondération spécifique qui assure la propriété de conservation (condition 3).

### Décomposition en ondes

Les valeurs propres de la matrice de Roe sont :

$$\tilde{\lambda}_1 = \tilde{u} - \tilde{a}, \qquad \tilde{\lambda}_2 = \tilde{u}, \qquad \tilde{\lambda}_3 = \tilde{u} + \tilde{a}$$

Le saut $\Delta \mathbf{U} = \mathbf{U}_R - \mathbf{U}_L$ est décomposé sur les vecteurs propres droits $\tilde{\mathbf{r}}_k$ :

$$\mathbf{U}_R - \mathbf{U}_L = \sum_{k=1}^{3} \tilde{\alpha}_k \tilde{\mathbf{r}}_k$$

où les **amplitudes d'onde** $\tilde{\alpha}_k$ mesurent l'intensité de chaque famille d'ondes :

$$\tilde{\alpha}_1 = \frac{\Delta p - \tilde{\rho}\, \tilde{a}\, \Delta u}{2\tilde{a}^2}$$

$$\tilde{\alpha}_2 = \Delta \rho - \frac{\Delta p}{\tilde{a}^2}$$

$$\tilde{\alpha}_3 = \frac{\Delta p + \tilde{\rho}\, \tilde{a}\, \Delta u}{2\tilde{a}^2}$$

avec $\tilde{\rho} = \sqrt{\rho_L \rho_R}$ (moyenne géométrique), $\Delta \rho = \rho_R - \rho_L$, $\Delta u = u_R - u_L$, $\Delta p = p_R - p_L$.

**Interprétation physique des amplitudes** :

- $\tilde{\alpha}_1$ et $\tilde{\alpha}_3$ : amplitudes des **ondes acoustiques** (gauche et droite). Elles sont non nulles quand il y a un saut de pression ($\Delta p \neq 0$) ou de vitesse ($\Delta u \neq 0$). La combinaison $\Delta p \pm \tilde{\rho} \tilde{a} \Delta u$ correspond aux invariants de Riemann du système linéarisé.

- $\tilde{\alpha}_2$ : amplitude de l'**onde entropique** (discontinuité de contact). Elle mesure la variation de densité non expliquée par les variations de pression : $\Delta \rho - \Delta p / \tilde{a}^2$. Si $\Delta \rho = \Delta p / \tilde{a}^2$ (relation isentropique), il n'y a pas d'onde de contact.

### Formule du flux

$$\hat{\mathbf{F}}_{i+1/2} = \frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \frac{1}{2} \sum_{k=1}^{3} |\tilde{\lambda}_k|\, \tilde{\alpha}_k\, \tilde{\mathbf{r}}_k$$

**Interprétation** :

- $\frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R)$ : flux centré, comme pour Rusanov (§4.2).

- $-\frac{1}{2} \sum_{k=1}^{3} |\tilde{\lambda}_k|\, \tilde{\alpha}_k\, \tilde{\mathbf{r}}_k$ : **terme de dissipation numérique**. Contrairement à Rusanov qui utilise une seule vitesse $S_{\max}$ pour les trois ondes, Roe dissipe chaque onde $k$ avec sa propre vitesse $|\tilde{\lambda}_k|$ et sa propre amplitude $\tilde{\alpha}_k$. C'est cette dissipation **sélective** qui rend Roe beaucoup plus précis.

En développant le produit $|\tilde{\lambda}_k|\, \tilde{\alpha}_k\, \tilde{\mathbf{r}}_k$ pour les trois ondes, on obtient la dissipation composante par composante (masse, quantité de mouvement, énergie) telle qu'implémentée dans le code.

### Correction entropique de Harten

Le schéma de Roe exact (sans correction) peut produire des **chocs d'expansion non physiques** (ou « expansion shocks »). Ce problème survient quand une valeur propre $\tilde{\lambda}_k$ passe par zéro : le terme $|\tilde{\lambda}_k|$ s'annule, la dissipation disparaît, et le schéma ne peut plus distinguer un choc physique (admissible) d'un choc d'expansion (non admissible au sens entropique).

La correction entropique de Harten-Hyman remplace $|\tilde{\lambda}_k|$ par une version régularisée :

$$|\tilde{\lambda}_k|_{\text{corrigé}} = \begin{cases} \varepsilon_k & \text{si } |\tilde{\lambda}_k| < \varepsilon_k \\ |\tilde{\lambda}_k| & \text{sinon} \end{cases}$$

avec le seuil :

$$\varepsilon_k = \max\bigl(0,\; \tilde{\lambda}_k - \lambda_k^L,\; \lambda_k^R - \tilde{\lambda}_k\bigr)$$

où $\lambda_k^L$ et $\lambda_k^R$ sont les valeurs propres évaluées aux états gauche et droit. Ce seuil est non nul uniquement quand la valeur propre change de signe entre les deux côtés (cas d'une détente transsonique), ce qui est exactement la situation où un choc d'expansion peut apparaître.

**Effet physique** : la correction ajoute un minimum de dissipation dans les zones de détente transsonique, empêchant la formation de chocs d'expansion. Elle n'affecte pas les ondes dont la vitesse ne s'annule pas.

### Avantages

- **Dissipation sélective** : chaque onde est dissipée à son propre taux. En particulier, l'onde de contact est dissipée uniquement au taux $|\tilde{u}|$, et non $|\tilde{u}| + \tilde{a}$ comme pour Rusanov.
- **Capture exacte des chocs isolés** : grâce à la propriété de conservation de la matrice de Roe, un choc séparant deux états constants est capturé en exactement un point (sans correction entropique).
- **Résolution des trois ondes** : comme HLLC, Roe distingue les trois familles d'ondes.
- **Vectorisé** : toutes les opérations sont vectorisées (pas de boucle Python).

### Inconvénients

- **Chocs d'expansion** : sans la correction entropique de Harten, le schéma peut produire des solutions non physiques dans les détentes transsoniques.
- **Coût modéré** : plus d'opérations par interface que HLL ou Rusanov (calcul des moyennes de Roe, décomposition en ondes, correction entropique), mais comparable à HLLC.
- **Non positivement conservatif** : contrairement à HLLC, Roe ne garantit pas *a priori* la positivité de la densité et de la pression. Dans des cas extrêmes (très faibles densités), des valeurs négatives peuvent apparaître.

### Comparaison

Roe et HLLC (§4.4) sont les deux schémas les plus utilisés en pratique. Ils résolvent tous deux les trois ondes et offrent une qualité de flux comparable. Les différences principales :

- **Robustesse** : HLLC est plus robuste (positivité garantie), Roe nécessite une correction entropique.
- **Approche** : HLLC simplifie la *structure* (quatre états avec des formules algébriques simples), Roe linéarise les *équations* (décomposition spectrale).
- **Capture des chocs** : Roe (sans correction) capture les chocs isolés exactement, HLLC non.

Par rapport à Godunov (§4.1), Roe offre une qualité comparable pour un coût bien moindre (pas d'itération de Newton).

### Références

- [Roe, 1981] P.L. Roe, « Approximate Riemann solvers, parameter vectors, and difference schemes », *J. Comput. Phys.*, 43(2), pp. 357-372.
- [Harten, Hyman, 1983] A. Harten, J.M. Hyman, « Self adjusting grid methods for one-dimensional hyperbolic conservation laws », *J. Comput. Phys.*, 50, pp. 235-269.

---

## §4.6 Schéma de Lax-Friedrichs (global)

### Motivation

Le schéma de Lax-Friedrichs est historiquement l'un des premiers schémas conservatifs pour les lois de conservation hyperboliques. Il ne repose pas sur un solveur de Riemann : c'est un **schéma centré** avec une dissipation numérique globale proportionnelle à $\Delta x / \Delta t$.

### Principe

Comme les schémas décentrés (§4.1–§4.5), le flux de Lax-Friedrichs a la forme « flux centré + dissipation ». Mais ici, le coefficient de dissipation n'est pas une vitesse d'onde locale : c'est le rapport global $\Delta x / \Delta t$, qui borne toujours la vitesse maximale (par la condition CFL : $\Delta t \leq \text{CFL} \cdot \Delta x / \max(|u| + a)$).

### Formule

$$\hat{\mathbf{F}}_{i+1/2} = \frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \frac{1}{2} \frac{\Delta x}{\Delta t} (\mathbf{U}_R - \mathbf{U}_L)$$

**Interprétation terme par terme** :

- $\frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R)$ : flux centré.

- $-\frac{1}{2} (\Delta x / \Delta t) (\mathbf{U}_R - \mathbf{U}_L)$ : dissipation numérique. Le coefficient $\Delta x / \Delta t$ est **global** : il est le même pour toutes les interfaces et pour toutes les ondes. Comme $\Delta x / \Delta t \geq \max(|u| + a) / \text{CFL}$, ce coefficient est toujours supérieur ou égal à la vitesse d'onde maximale. La dissipation est donc *au moins* aussi forte que celle de Rusanov, et en pratique beaucoup plus forte (sauf si CFL = 1 et la vitesse maximale est atteinte partout).

### Comparaison avec Rusanov

La différence entre Lax-Friedrichs et Rusanov (§4.2) est subtile mais importante :

| | Lax-Friedrichs | Rusanov |
|---|---|---|
| Coefficient de dissipation | $\Delta x / \Delta t$ (global) | $S_{\max} = \max(\|u_L\| + a_L, \|u_R\| + a_R)$ (local) |
| Dépend du pas de temps | Oui | Non |
| Type | Centré | Décentré (upwind) |
| Dissipation | Maximale | Forte mais locale |

Rusanov adapte sa dissipation aux conditions locales (là où les ondes sont rapides, la dissipation est plus forte) tandis que Lax-Friedrichs utilise le même coefficient partout.

### Avantages

- **Extrême simplicité** : la formule ne fait intervenir que les flux physiques et les variables conservatives, sans aucun calcul de vitesse d'onde.
- **Stabilité inconditionnelle** (sous condition CFL) : la forte dissipation garantit la stabilité.

### Inconvénients

- **Schéma le plus dissipatif** : la dissipation est maximale pour toute méthode stable sous condition CFL. Les discontinuités (chocs et contacts) sont fortement étalées.
- **Non composable** : ne peut pas être combiné avec les reconstructions d'ordre élevé (pas de solveur de Riemann).
- **Dépendance au pas de temps** : la dissipation dépend de $\Delta t$, ce qui couple le schéma spatial au choix du pas de temps.

### Références

- [Lax, 1954] P.D. Lax, « Weak solutions of nonlinear hyperbolic equations and their numerical computation », *Comm. Pure Appl. Math.*, 7, pp. 159-193.

---

## §4.7 Schéma de Lax-Wendroff (Richtmyer deux étapes)

### Motivation

Le schéma de Lax-Friedrichs (§4.6) est d'ordre 1 et très dissipatif. Le schéma de Lax-Wendroff vise l'**ordre 2 en espace et en temps** tout en restant un schéma centré, sans solveur de Riemann. L'idée est d'utiliser un prédicteur de type Lax-Friedrichs pour estimer l'état à mi-pas de temps, puis d'évaluer le flux à partir de cet état prédit.

### Principe

Le schéma de Lax-Wendroff dans sa variante de Richtmyer est un schéma **prédicteur-correcteur** en deux étapes :

1. **Prédicteur** : on estime l'état au milieu de l'interface et au milieu du pas de temps par une demi-étape de type Lax-Friedrichs.
2. **Correcteur** : on évalue le flux physique à partir de cet état prédit.

### Formule

**Prédicteur** (état au demi-pas de temps à l'interface) :

$$\mathbf{U}^{n+1/2}_{i+1/2} = \frac{1}{2}(\mathbf{U}_L + \mathbf{U}_R) - \frac{1}{2} \frac{\Delta t}{\Delta x}(\mathbf{F}_R - \mathbf{F}_L)$$

**Correcteur** (flux numérique) :

$$\hat{\mathbf{F}}_{i+1/2} = \mathbf{F}\bigl(\mathbf{U}^{n+1/2}_{i+1/2}\bigr)$$

**Interprétation du prédicteur** :

- $\frac{1}{2}(\mathbf{U}_L + \mathbf{U}_R)$ : moyenne des états de part et d'autre de l'interface. C'est l'approximation la plus simple de l'état à l'interface.

- $-\frac{1}{2} (\Delta t / \Delta x)(\mathbf{F}_R - \mathbf{F}_L)$ : correction temporelle. Ce terme avance la solution d'un demi-pas de temps en utilisant la différence des flux physiques. C'est un schéma d'Euler explicite sur un demi-pas de temps.

**Interprétation du correcteur** :

Le flux final est simplement le flux physique évalué à l'état prédit. Comme cet état incorpore déjà l'information temporelle (demi-pas de temps), le schéma atteint l'ordre 2 en temps sans nécessiter un intégrateur de Runge-Kutta multi-étages.

### Propriété spéciale : `time_integral_included = True`

Le prédicteur intègre déjà un demi-pas de temps dans le calcul du flux. L'intégration temporelle est donc **incluse** dans le flux. Si l'on utilisait un intégrateur RK2 ou RK3 en plus, on avancerait la solution de manière incorrecte. C'est pourquoi le code force l'utilisation de RK1 (Euler explicite) quand Lax-Wendroff est sélectionné.

### Avantages

- **Ordre 2 en espace et en temps** : c'est le seul schéma de ce chapitre qui est intrinsèquement d'ordre 2 dans les deux dimensions (les autres sont d'ordre 1 en espace et nécessitent une reconstruction d'ordre élevé pour monter en ordre spatial).
- **Simplicité** : pas de solveur de Riemann, pas de décomposition en ondes. Deux évaluations du flux physique suffisent.
- **Pas d'intégrateur temporel supplémentaire** : l'intégration temporelle est intégrée, ce qui simplifie l'algorithme.

### Inconvénients

- **Oscillations dispersives** : comme tout schéma d'ordre 2 sans limitation, Lax-Wendroff produit des oscillations parasites (oscillations de Gibbs) au voisinage des discontinuités. Ces oscillations sont de nature **dispersive** : les hautes fréquences numériques se propagent à des vitesses différentes de la vitesse physique.
- **Non composable** : la formulation prédicteur-correcteur ne se décompose pas en « reconstruction + solveur de Riemann ». On ne peut pas le combiner avec MUSCL ou WENO.
- **Sensibilité aux chocs** : en l'absence de limiteur ou de viscosité artificielle, les oscillations rendent le schéma peu fiable pour les problèmes avec des chocs forts.

### Comparaison

Lax-Wendroff est complémentaire des schémas décentrés. Là où Rusanov (§4.2) ou HLL (§4.3) sont trop dissipatifs, Lax-Wendroff est trop dispersif. Le schéma JST (§4.8) peut être vu comme une amélioration de l'idée centrée : il ajoute une dissipation artificielle adaptative pour contrôler les oscillations tout en préservant la précision en zone lisse.

### Références

- [Lax, Wendroff, 1960] P.D. Lax, B. Wendroff, « Systems of conservation laws », *Comm. Pure Appl. Math.*, 13, pp. 217-237.
- [Richtmyer, 1963] R.D. Richtmyer, « A survey of difference methods for non-steady fluid dynamics », *NCAR Technical Note*, 63-2.

---

## §4.8 Schéma JST

### Motivation

Les schémas centrés comme Lax-Wendroff (§4.7) sont d'ordre 2 en zone lisse mais oscillent près des chocs. Les schémas décentrés (§4.1–§4.5) capturent bien les chocs mais nécessitent un solveur de Riemann. Le schéma JST (Jameson, Schmidt, Turkel) propose un compromis : un **flux centré** avec une **dissipation artificielle adaptative** qui s'active automatiquement près des chocs et disparaît en zone lisse.

### Principe

Le flux JST est la somme de trois contributions :

$$\hat{\mathbf{F}}_{i+1/2} = \underbrace{\frac{1}{2}(\mathbf{F}_i + \mathbf{F}_{i+1})}_{\text{flux centré}} - \underbrace{\mathbf{d}^{(2)}_{i+1/2}}_{\text{dissipation d'ordre 2}} + \underbrace{\mathbf{d}^{(4)}_{i+1/2}}_{\text{dissipation d'ordre 4}}$$

L'idée maîtresse est d'utiliser un **capteur de choc** basé sur la pression pour basculer automatiquement entre deux régimes :

- **Près d'un choc** : la dissipation d'ordre 2 ($\mathbf{d}^{(2)}$) est activée. Elle est similaire à celle de Rusanov et empêche les oscillations.
- **En zone lisse** : la dissipation d'ordre 2 est désactivée et la dissipation d'ordre 4 ($\mathbf{d}^{(4)}$) prend le relais. Cette dissipation de fond, plus faible, lisse uniquement les modes numériques parasites de haute fréquence sans affecter la solution physique.

### Capteur de choc

Le capteur de choc est un indicateur adimensionnel basé sur la courbure de la pression :

$$\nu_i = \frac{|p_{i+1} - 2p_i + p_{i-1}|}{p_{i+1} + 2p_i + p_{i-1}}$$

**Interprétation** :

- Le **numérateur** $|p_{i+1} - 2p_i + p_{i-1}|$ est la valeur absolue de la différence seconde de la pression. Il est grand en présence d'un choc (forte variation de pression) et petit en zone lisse (pression variant linéairement ou quadratiquement).

- Le **dénominateur** $p_{i+1} + 2p_i + p_{i-1}$ normalise l'indicateur par le niveau moyen de pression, rendant $\nu_i$ adimensionnel et indépendant de l'amplitude absolue.

- $\nu_i \approx 0$ en zone lisse (la différence seconde est d'ordre $\Delta x^2$ tandis que la somme est d'ordre $p$).
- $\nu_i \approx O(1)$ près d'un choc.

### Coefficients de dissipation

Les coefficients de dissipation à l'interface $i+1/2$ sont calculés à partir du capteur :

$$\varepsilon^{(2)}_{i+1/2} = \kappa_2 \cdot \max(\nu_{i-1}, \nu_i, \nu_{i+1}, \nu_{i+2})$$

$$\varepsilon^{(4)}_{i+1/2} = \max\bigl(0,\; \kappa_4 - \varepsilon^{(2)}_{i+1/2}\bigr)$$

**Interprétation** :

- $\varepsilon^{(2)}$ est proportionnel au maximum du capteur de choc dans le voisinage de l'interface. L'opérateur $\max$ sur 4 voisins assure que la dissipation d'ordre 2 est active dans tout le voisinage du choc, pas seulement au point de discontinuité.

- $\varepsilon^{(4)}$ est le complément : quand $\varepsilon^{(2)}$ est grand (choc), $\varepsilon^{(4)}$ est nul. Quand $\varepsilon^{(2)}$ est nul (zone lisse), $\varepsilon^{(4)} = \kappa_4$, et la dissipation d'ordre 4 est à son niveau maximal.

- **Paramètres** : $\kappa_2 = 1$ et $\kappa_4 = 1/64$ sont les valeurs par défaut. $\kappa_2$ contrôle l'intensité de la dissipation près des chocs, $\kappa_4$ contrôle le lissage de fond.

### Termes de dissipation

Les termes de dissipation font intervenir le rayon spectral local $\lambda_{i+1/2} = \frac{1}{2}(|u_i| + a_i + |u_{i+1}| + a_{i+1})$ :

**Dissipation d'ordre 2** (active près des chocs) :

$$\mathbf{d}^{(2)}_{i+1/2} = \varepsilon^{(2)}_{i+1/2} \cdot \lambda_{i+1/2} \cdot (\mathbf{U}_{i+1} - \mathbf{U}_i)$$

C'est un terme de type Rusanov local, proportionnel au saut de variables conservatives. Il ajoute une diffusion numérique du premier ordre au voisinage des chocs, suffisante pour supprimer les oscillations.

**Dissipation d'ordre 4** (active en zone lisse) :

$$\mathbf{d}^{(4)}_{i+1/2} = \varepsilon^{(4)}_{i+1/2} \cdot \lambda_{i+1/2} \cdot (\mathbf{U}_{i+2} - 3\mathbf{U}_{i+1} + 3\mathbf{U}_i - \mathbf{U}_{i-1})$$

C'est un terme de type biharmonique (différence quatrième), analogue à une hyperdiffusion $-\nu \cdot \partial^4 \mathbf{U} / \partial x^4$. Il atténue sélectivement les modes de haute fréquence (oscillations numériques) tout en préservant les structures à basse fréquence (solution physique). Le stencil à 4 points ($i-1, i, i+1, i+2$) est la raison pour laquelle JST nécessite $n_{\text{ghost}} = 2$.

### Intégration temporelle

JST utilise par défaut un intégrateur **RK4** (Runge-Kutta classique à 4 étages). Ce choix est historique : le schéma JST a été conçu dans le contexte de la convergence vers un état stationnaire, où RK4 agit comme un lisseur itératif efficace. Pour les problèmes instationnaires de ce code, RK4 est aussi un bon choix car il fournit une précision temporelle d'ordre 4, bien supérieure à ce que l'ordre spatial (2) exigerait.

### Avantages

- **Adaptatif** : la dissipation s'ajuste automatiquement à la solution. Pas de paramètre à régler manuellement (les valeurs par défaut de $\kappa_2$ et $\kappa_4$ conviennent dans la plupart des cas).
- **Pas de solveur de Riemann** : le flux est entièrement centré, ce qui simplifie l'extension à des systèmes d'équations plus complexes ou à plusieurs dimensions.
- **Ordre 2 effectif** : en zone lisse, la dissipation de fond est d'ordre 4, ce qui ne dégrade pas l'ordre 2 du flux centré.
- **Performances** : toutes les opérations sont vectorisées.

### Inconvénients

- **Non composable** : comme Lax-Friedrichs et Lax-Wendroff, JST ne se décompose pas en « reconstruction + solveur de Riemann ».
- **Réglage des paramètres** : bien que les valeurs par défaut fonctionnent bien, des cas extrêmes peuvent nécessiter un ajustement de $\kappa_2$ et $\kappa_4$.
- **Stencil large** : le terme d'ordre 4 nécessite 4 cellules ($i-1$ à $i+2$), d'où $n_{\text{ghost}} = 2$.

### Références

- [Jameson, Schmidt, Turkel, 1981] A. Jameson, W. Schmidt, E. Turkel, « Numerical solutions of the Euler equations by finite volume methods using Runge-Kutta time stepping schemes », *AIAA Paper*, 81-1259.

---

## §4.9 Schéma AUSM+ (*Advection Upstream Splitting Method*)

### Motivation

Les solveurs de Riemann (Godunov, Rusanov, HLL, HLLC, Roe) approchent la solution du problème de Riemann local pour calculer le flux numérique. Le schéma AUSM+ de Liou [Liou, 1996] adopte une approche radicalement différente : il **sépare le flux physique** en deux contributions de nature distincte — une partie **convective** (transport de masse, quantité de mouvement et énergie par le débit massique) et une partie **pression** (force de pression) — et traite chacune avec son propre *splitting* basé sur le nombre de Mach local.

### Décomposition du flux

Le flux d'Euler $\mathbf{F} = (\rho u, \rho u^2 + p, u(E + p))^T$ se décompose naturellement en :

$$\mathbf{F} = \dot{m} \, \boldsymbol{\Phi} + \mathbf{P}$$

avec :

- $\dot{m} = \rho u$ le **débit massique**,
- $\boldsymbol{\Phi} = (1, u, H)^T$ les **quantités transportées** (1, vitesse, enthalpie totale),
- $\mathbf{P} = (0, p, 0)^T$ le **flux de pression**.

### Nombre de Mach d'interface

La vitesse du son à l'interface est prise comme la moyenne arithmétique :

$$a_{1/2} = \frac{a_L + a_R}{2}$$

Les nombres de Mach gauche et droit sont :

$$M_L = \frac{u_L}{a_{1/2}}, \qquad M_R = \frac{u_R}{a_{1/2}}$$

Le nombre de Mach d'interface est la combinaison des splittings :

$$M_{1/2} = \mathcal{M}^+(M_L) + \mathcal{M}^-(M_R)$$

### Fonctions de splitting

Les fonctions de splitting du Mach sont des polynômes de degré 4 qui assurent une transition lisse entre les régimes subsonique ($|M| < 1$) et supersonique ($|M| \geq 1$) :

$$\mathcal{M}^{\pm}(M) = \begin{cases} \frac{1}{2}(M \pm |M|) & \text{si } |M| \geq 1 \\ \pm\frac{1}{4}(M \pm 1)^2 \pm \frac{1}{8}(M^2 - 1)^2 & \text{si } |M| < 1 \end{cases}$$

Les fonctions de splitting de la pression sont :

$$\mathcal{P}^{\pm}(M) = \begin{cases} \frac{1}{2}(1 \pm \text{sign}(M)) & \text{si } |M| \geq 1 \\ \frac{1}{4}(M \pm 1)^2(2 \mp M) \pm \frac{3}{16}M(M^2 - 1)^2 & \text{si } |M| < 1 \end{cases}$$

### Formule du flux

Le flux numérique complet à l'interface est :

$$\hat{\mathbf{F}}_{1/2} = \dot{m}_{1/2} \, \boldsymbol{\Phi}_{1/2} + \mathbf{P}_{1/2}$$

avec le débit massique :

$$\dot{m}_{1/2} = a_{1/2} \, M_{1/2} \times \begin{cases} \rho_L & \text{si } M_{1/2} \geq 0 \\ \rho_R & \text{sinon} \end{cases}$$

Les quantités transportées sont sélectionnées par upwinding sur le signe de $\dot{m}_{1/2}$ :

$$\boldsymbol{\Phi}_{1/2} = \begin{cases} (1, u_L, H_L)^T & \text{si } \dot{m}_{1/2} \geq 0 \\ (1, u_R, H_R)^T & \text{sinon} \end{cases}$$

La pression d'interface est :

$$p_{1/2} = \mathcal{P}^+(M_L) \, p_L + \mathcal{P}^-(M_R) \, p_R$$

### Avantages

- **Résolution exacte des contacts stationnaires** : quand $u = 0$, le débit massique $\dot{m} = 0$ et le flux convectif s'annule. La pression d'interface vaut $\mathcal{P}^+(0) p_L + \mathcal{P}^-(0) p_R = p/2 + p/2 = p$ si $p_L = p_R$. Aucune diffusion numérique.
- **Pas de correction entropique** nécessaire (les fonctions de splitting sont intrinsèquement bien conditionnées).
- **Flux lisse** : les polynômes de splitting sont continûment différentiables, contrairement aux formules conditionnelles des solveurs de Riemann.
- **Composable** avec les reconstructions d'ordre élevé (MUSCL, ENO, WENO).

### Inconvénients

- **Précision légèrement inférieure** à Roe sur les chocs isolés (pas de propriété de Rankine-Hugoniot discrète comme Roe).
- **Sensibilité basse vitesse** : en régime quasi-incompressible ($M \ll 1$), les oscillations de pression « *checkerboard* » peuvent apparaître. La variante AUSM+-up corrige ce défaut.

### Comparaison

AUSM+ se situe entre HLLC et Rusanov en termes de dissipation sur les cas test classiques. Il est particulièrement performant sur les écoulements à vitesse modérée et les contacts. Sa philosophie de splitting le rend très différent des solveurs de Riemann et offre une perspective complémentaire.

### Références

- [Liou, 1996] M.-S. Liou, « A sequel to AUSM: AUSM+ », *J. Comput. Phys.*, 129, pp. 364-382.

---

## §4.10 Tableau comparatif

Le tableau suivant résume les caractéristiques des huit schémas de flux :

| Schéma | Type | Ondes résolues | Ordre spatial | Composable | Coût relatif | Dissipation | Cas d'usage recommandé |
|--------|------|---------------|---------------|------------|-------------|-------------|----------------------|
| Godunov (§4.1) | Upwind | 3 (exact) | 1 | Oui | Élevé | Optimale | Référence |
| Rusanov (§4.2) | Upwind | 1 | 1 | Oui | Faible | Très forte | Prototypage, cas difficiles |
| HLL (§4.3) | Upwind | 2 | 1 | Oui | Faible | Forte | Compromis simple |
| HLLC (§4.4) | Upwind | 3 | 1 | Oui | Moyen | Modérée | **Usage général** |
| Roe (§4.5) | Upwind | 3 | 1 | Oui | Moyen | Modérée | Précision maximale |
| Roe sans correction (§4.5) | Upwind | 3 | 1 | Oui | Moyen | Modérée | Comparaison pédagogique |
| AUSM+ (§4.9) | FVS | — | 1 | Oui | Moyen | Modérée | Écoulements multi-vitesse |
| AUSM+-up (§4.9) | FVS | — | 1 | Oui | Moyen | Modérée | AUSM+ amélioré, moins d'oscillations |
| Lax-Friedrichs (§4.6) | Centré | — | 1 | Non | Très faible | Maximale | Pédagogique |
| Lax-Wendroff (§4.7) | Centré | — | 2 | Non | Faible | Faible (dispersif) | Solutions lisses |
| JST (§4.8) | Centré | — | 2 | Non | Moyen | Adaptative | Aérodynamique |

### Lecture du tableau

- **Ondes résolues** : nombre de familles d'ondes distinguées par le schéma. Plus le nombre est élevé, meilleure est la résolution des discontinuités. Les schémas centrés n'utilisent pas de solveur de Riemann et ne résolvent pas les ondes au sens strict.

- **Composable** : indique si le flux peut être combiné avec les reconstructions d'ordre élevé (MUSCL, ENO, WENO). Les 7 flux composables (Godunov, Rusanov, HLL, HLLC, Roe, Roe sans correction, AUSM+) combinés avec 6 reconstructions donnent 42 schémas d'ordre élevé (cf. §3.7).

- **Coût relatif** : estimation qualitative du coût par interface. Godunov est le plus cher (itération de Newton). Les schémas vectorisés (Rusanov, HLL, Lax-Friedrichs) sont les moins chers. HLLC est plus cher dans l'implémentation actuelle (boucle Python) mais pourrait être vectorisé.

- **Dissipation** : niveau de dissipation numérique introduite par le schéma. Les schémas les plus dissipatifs étalent les discontinuités mais sont plus robustes. Les schémas les moins dissipatifs préservent mieux les discontinuités mais peuvent osciller (Lax-Wendroff) ou nécessiter des corrections (Roe).

### Recommandation pratique

Pour un usage général sur les équations d'Euler 1D :

1. **HLLC** est le meilleur choix en combinaison avec une reconstruction d'ordre élevé (MUSCL-HLLC, WENO3-HLLC, WENO5-HLLC). Il offre le meilleur rapport qualité/coût.
2. **Roe** est une alternative quand on souhaite une capture optimale des chocs isolés, au prix d'une moindre robustesse.
3. **Rusanov** est utile pour le prototypage ou quand la robustesse prime sur la précision.
4. **JST** est intéressant pour l'aérodynamique stationnaire ou quand on souhaite éviter un solveur de Riemann.

---

## §4.11 Implémentation

Le code correspondant à ce chapitre se trouve dans le répertoire `euler1d/schemes/flux/` :

| Fichier | Schéma | Section |
|---------|--------|---------|
| [`godunov.py`](../euler1d/schemes/flux/godunov.py) | Godunov | §4.1 |
| [`rusanov.py`](../euler1d/schemes/flux/rusanov.py) | Rusanov | §4.2 |
| [`hll.py`](../euler1d/schemes/flux/hll.py) | HLL | §4.3 |
| [`hllc.py`](../euler1d/schemes/flux/hllc.py) | HLLC | §4.4 |
| [`roe.py`](../euler1d/schemes/flux/roe.py) | Roe (avec et sans correction entropique) | §4.5 |
| [`ausm_plus.py`](../euler1d/schemes/flux/ausm_plus.py) | AUSM+ | §4.9 |
| [`lax_friedrichs.py`](../euler1d/schemes/flux/lax_friedrichs.py) | Lax-Friedrichs | §4.6 |
| [`lax_wendroff.py`](../euler1d/schemes/flux/lax_wendroff.py) | Lax-Wendroff | §4.7 |
| [`jst.py`](../euler1d/schemes/flux/jst.py) | JST | §4.8 |
| [`__init__.py`](../euler1d/schemes/flux/__init__.py) | Registre des flux composables | — |

Le registre `__init__.py` expose la fonction `get_flux_solver(name)` qui instancie les sept flux composables (Rusanov, HLL, HLLC, Roe, Roe-NC, Godunov, AUSM+) par nom. Les trois schémas centrés non composables (Lax-Friedrichs, Lax-Wendroff, JST) sont enregistrés directement dans le registre principal `euler1d/schemes/__init__.py` (cf. §3.7).

---

## Références

- [Batten et al., 1997] P. Batten, N. Clarke, C. Lambert, D.M. Causon, « On the choice of wavespeeds for the HLLC Riemann solver », *SIAM J. Sci. Comput.*, 18(6), pp. 1553-1570.
- [Davis, 1988] S.F. Davis, « Simplified second-order Godunov-type methods », *SIAM J. Sci. Stat. Comput.*, 9(3), pp. 445-473.
- [Godunov, 1959] S.K. Godunov, « A difference method for numerical calculation of discontinuous solutions of the equations of hydrodynamics », *Mat. Sb.*, 47(3), pp. 271-306.
- [Harten, Hyman, 1983] A. Harten, J.M. Hyman, « Self adjusting grid methods for one-dimensional hyperbolic conservation laws », *J. Comput. Phys.*, 50, pp. 235-269.
- [Harten, Lax, van Leer, 1983] A. Harten, P.D. Lax, B. van Leer, « On upstream differencing and Godunov-type schemes for hyperbolic conservation laws », *SIAM Review*, 25(1), pp. 35-61.
- [Jameson, Schmidt, Turkel, 1981] A. Jameson, W. Schmidt, E. Turkel, « Numerical solutions of the Euler equations by finite volume methods using Runge-Kutta time stepping schemes », *AIAA Paper*, 81-1259.
- [Lax, 1954] P.D. Lax, « Weak solutions of nonlinear hyperbolic equations and their numerical computation », *Comm. Pure Appl. Math.*, 7, pp. 159-193.
- [Lax, Wendroff, 1960] P.D. Lax, B. Wendroff, « Systems of conservation laws », *Comm. Pure Appl. Math.*, 13, pp. 217-237.
- [Richtmyer, 1963] R.D. Richtmyer, « A survey of difference methods for non-steady fluid dynamics », *NCAR Technical Note*, 63-2.
- [Roe, 1981] P.L. Roe, « Approximate Riemann solvers, parameter vectors, and difference schemes », *J. Comput. Phys.*, 43(2), pp. 357-372.
- [Rusanov, 1961] V.V. Rusanov, « The calculation of the interaction of non-stationary shock waves with barriers », *Zh. Vychisl. Mat. i Mat. Fiz.*, 1(2), pp. 267-279.
- [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e éd., Springer.
