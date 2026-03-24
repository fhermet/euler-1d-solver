# Chapitre 5 : Reconstructions d'ordre élevé

> **Prérequis** : [Chapitre 3 — Méthode des volumes finis](03_finite_volume.md) (semi-discrétisation, cellules fantômes, principe de composition reconstruction + flux), [Chapitre 4 — Flux numériques](04_flux_schemes.md) (solveurs de Riemann composables).

## Introduction

Un schéma de volumes finis d'ordre 1 utilise une reconstruction **constante par morceaux** : dans chaque cellule $i$, la solution approchée est simplement la valeur moyenne $\mathbf{W}_i$. Cette approximation rudimentaire introduit une **diffusion numérique excessive** qui lisse les gradients, étale les discontinuités de contact sur de nombreuses cellules et amortit les ondes acoustiques sur de longues distances de propagation. Pour des applications réalistes — capture fine des chocs, propagation d'ondes sur de longs domaines, simulations instationnaires précises — l'ordre 1 est insuffisant.

Peut-on simplement utiliser une reconstruction linéaire (ou polynomiale de degré supérieur) sans précaution particulière ? **Non.** Le **théorème de Godunov** (1959) affirme :

> *Un schéma linéaire monotone (ne créant pas de nouveaux extrema) est au plus d'ordre 1.*

Ce résultat fondamental signifie que toute montée en ordre passe nécessairement par une reconstruction **non linéaire**, c'est-à-dire dont les coefficients dépendent de la solution locale. C'est exactement ce que font les méthodes MUSCL (limiteurs de pente), ENO (sélection de stencil) et WENO (pondération non linéaire de stencils) présentées dans ce chapitre.

### Reconstruction sur les variables primitives

Toutes les reconstructions de ce solveur opèrent sur les **variables primitives** $\mathbf{W} = (\rho, u, p)$ et non sur les variables conservatives $\mathbf{U} = (\rho, \rho u, E)$. Ce choix est motivé par le meilleur **conditionnement** des variables primitives au voisinage des discontinuités :

- Près d'un **choc**, $\rho u$ et $E$ présentent des sauts couplés (le produit $\rho u$ mélange densité et vitesse), tandis que $\rho$, $u$ et $p$ ont des sauts indépendants et plus réguliers.
- Près d'une **discontinuité de contact**, seule $\rho$ est discontinue parmi les primitives, alors que $\rho u$ et $E$ le sont aussi parmi les conservatives.
- Les limiteurs et indicateurs de régularité fonctionnent mieux sur des variables dont les variations sont découplées et monotones par morceaux.

Le flux de calcul est donc : $\mathbf{U} \to \mathbf{W}$ (conversion), reconstruction de $\mathbf{W}_L, \mathbf{W}_R$ aux interfaces, puis $\mathbf{W}_L, \mathbf{W}_R \to \mathbf{U}_L, \mathbf{U}_R$ (reconversion) avant passage au solveur de Riemann.

### Principe de composition

Un schéma d'ordre élevé se compose de deux briques indépendantes (cf. §3.5) :

$$\text{Schéma d'ordre élevé} = \underbrace{\text{Reconstruction spatiale}}_{\text{MUSCL, ENO, WENO}} + \underbrace{\text{Flux numérique}}_{\text{Rusanov, HLL, HLLC, Roe, Godunov}}$$

La reconstruction fournit les états $\mathbf{W}_L$ et $\mathbf{W}_R$ à chaque interface ; le flux numérique les utilise pour calculer $\hat{\mathbf{F}}_{i+1/2}$. Toute combinaison reconstruction × flux composable est valide (cf. Chapitre 4, Introduction). Les schémas centrés (Lax-Friedrichs, Lax-Wendroff, JST) ne sont **pas** composables avec ces reconstructions.

### Plancher de positivité

Toutes les reconstructions de ce solveur appliquent un **plancher de positivité** aux valeurs reconstruites de $\rho$ et $p$ :

$$\rho_L, \rho_R \geq 10^{-10}, \qquad p_L, p_R \geq 10^{-10}$$

Ce garde-fou empêche l'apparition de densités ou pressions négatives qui provoqueraient des erreurs numériques (racine carrée d'un nombre négatif dans le calcul de la vitesse du son).

---

## §5.1 MUSCL + limiteurs de pente

### Motivation

La méthode **MUSCL** (*Monotone Upstream-centered Scheme for Conservation Laws*), introduite par van Leer [van Leer, 1979], est la plus répandue des reconstructions d'ordre 2. L'idée est simple : au lieu de représenter la solution par une constante dans chaque cellule, on utilise une **droite** (reconstruction affine). La pente de cette droite est **limitée** pour respecter une condition de type TVD (*Total Variation Diminishing*), ce qui empêche la création de nouveaux extrema.

### Reconstruction linéaire par morceaux

Dans chaque cellule $i$, la solution reconstruite est :

$$\mathbf{W}(x) = \mathbf{W}_i + \frac{x - x_i}{\Delta x} \, \boldsymbol{\sigma}_i$$

où $\boldsymbol{\sigma}_i$ est la pente limitée. Les états reconstruits aux interfaces sont :

$$\mathbf{W}_{i+1/2}^L = \mathbf{W}_i + \frac{1}{2} \boldsymbol{\sigma}_i, \qquad \mathbf{W}_{i+1/2}^R = \mathbf{W}_{i+1} - \frac{1}{2} \boldsymbol{\sigma}_{i+1}$$

### Calcul de la pente limitée

On définit les différences *forward* aux interfaces :

$$\Delta \mathbf{W}_{i+1/2} = \mathbf{W}_{i+1} - \mathbf{W}_i$$

Le **rapport de pente** dans la cellule $i$ est :

$$r_i = \frac{\Delta \mathbf{W}_{i-1/2}}{\Delta \mathbf{W}_{i+1/2}}$$

La pente limitée est alors :

$$\boldsymbol{\sigma}_i = \varphi(r_i) \times \Delta \mathbf{W}_{i+1/2}$$

où $\varphi(r)$ est la **fonction limiteur**. Le rôle du limiteur est de réduire la pente quand le rapport $r_i$ indique un changement de monotonie ou un extremum local.

### Région TVD de Sweby

Pour qu'un limiteur $\varphi(r)$ soit TVD, Sweby [Sweby, 1984] a montré qu'il doit satisfaire les conditions suivantes :

1. $\varphi(r) = 0$ pour $r \leq 0$ (annulation de la pente aux extrema).
2. La courbe $\varphi(r)$ doit se situer dans la **région de Sweby**, délimitée par :
   - **Borne inférieure** : le limiteur minmod, $\varphi(r) = \min(1, r)$.
   - **Borne supérieure** : le limiteur superbee, $\varphi(r) = \max(\min(2r, 1), \min(r, 2))$.
3. $\varphi(1) = 1$ (reconstruction centrée quand la solution est localement linéaire).

Tout limiteur dont le graphe reste dans cette région est TVD et d'ordre 2 en zone lisse.

### Limiteurs de pente

#### Minmod

$$\varphi(r) = \max\bigl(0,\, \min(1, r)\bigr)$$

- **Propriétés** : limiteur le plus dissipatif de la région TVD (borne inférieure de Sweby). Sélectionne la pente de plus petite valeur absolue entre la différence amont et la différence aval.
- **Avantages** : très robuste, jamais d'oscillation, convergence garantie. Idéal pour les problèmes raides.
- **Inconvénients** : trop dissipatif en zone lisse ; lisse excessivement les discontinuités de contact et les gradients doux. Réduit l'ordre effectif près des extrema lisses.

#### Van Leer

$$\varphi(r) = \frac{r + |r|}{1 + |r|}$$

- **Propriétés** : limiteur **différentiable** (lisse en $r = 0$), situé au milieu de la région de Sweby. Bon compromis entre dissipation et précision.
- **Avantages** : pas de discontinuité dans la fonction limiteur, ce qui favorise la convergence des solveurs implicites. Excellent compromis dissipation/précision pour la plupart des applications.
- **Inconvénients** : légèrement plus dissipatif que superbee ou MC sur les contacts.

#### Superbee

$$\varphi(r) = \max\bigl(0,\, \max(\min(2r, 1),\, \min(r, 2))\bigr)$$

- **Propriétés** : limiteur le **moins dissipatif** de la région TVD (borne supérieure de Sweby). Maximise la pente dans les limites TVD.
- **Avantages** : excellent pour les discontinuités de contact qu'il maintient très raides. Très faible dissipation numérique.
- **Inconvénients** : peut **artificiellement raidir** des profils lisses (effet de « crénelage » ou *staircase effect*), transformant des transitions douces en marches d'escalier. Ce comportement compressif est parfois indésirable.

#### MC (Monotonized Central)

$$\varphi(r) = \max\bigl(0,\, \min(2r,\, \tfrac{1+r}{2},\, 2)\bigr)$$

- **Propriétés** : utilise la pente **centrée** $(1+r)/2$ contrainte par les bornes TVD $2r$ et $2$. Se situe entre van Leer et superbee.
- **Avantages** : meilleure précision que van Leer en zone lisse grâce à la pente centrée. Bonne résolution des contacts sans l'effet compressif de superbee.
- **Inconvénients** : non différentiable (comme minmod et superbee), ce qui peut ralentir la convergence des schémas implicites.

#### Van Albada

$$\varphi(r) = \begin{cases} \dfrac{r^2 + r}{r^2 + 1} & \text{si } r > 0 \\ 0 & \text{sinon} \end{cases}$$

- **Propriétés** : limiteur **différentiable** utilisant un mélange quadratique rationnel. Comme van Leer, la transition est lisse en $r = 0$ et $r = 1$. Se situe dans la partie basse de la région de Sweby, entre minmod et van Leer.
- **Avantages** : continuité $C^1$ qui favorise la convergence des schémas implicites. Produit des solutions très régulières. Satisfait la propriété TVD.
- **Inconvénients** : plus dissipatif que van Leer, surtout près des discontinuités de contact. Les termes quadratiques $r^2$ au numérateur et dénominateur atténuent davantage les gradients modérés.

*Référence : [van Albada et al., 1982] G. D. van Albada, B. van Leer, W. W. Roberts, "A comparative study of computational methods in cosmic gas dynamics", Astronomy & Astrophysics, 108, pp. 76–84.*

### Tableau comparatif des limiteurs

| Limiteur | Formule | Dissipation | Différentiable | Position Sweby | Risque compressif |
|---|---|---|---|---|---|
| Minmod | $\max(0, \min(1, r))$ | Maximale | Non | Borne inférieure | Aucun |
| Van Albada | $(r^2+r)/(r^2+1)$ | Élevée | **Oui** | Bas (entre minmod et van Leer) | Aucun |
| Van Leer | $(r+\|r\|)/(1+\|r\|)$ | Modérée | **Oui** | Centre | Aucun |
| MC | $\max(0, \min(2r, \frac{1+r}{2}, 2))$ | Faible | Non | Centre-haut | Faible |
| Superbee | $\max(0, \max(\min(2r,1), \min(r,2)))$ | Minimale | Non | Borne supérieure | **Élevé** |

### Cellules fantômes et ordre

- **Ordre spatial** : 2.
- **Cellules fantômes** : $n_{\text{ghost}} = 2$ (une différence de chaque côté du stencil MUSCL).
- **Intégrateur temporel par défaut** : RK2 (Heun), cohérent avec l'ordre spatial 2.

### Références

- [van Leer, 1979] B. van Leer, *Towards the Ultimate Conservative Difference Scheme. V. A Second-Order Sequel to Godunov's Method*, J. Comput. Phys., 32, pp. 101–136.
- [Sweby, 1984] P. K. Sweby, *High Resolution Schemes Using Flux Limiters*, SIAM J. Numer. Anal., 21(5), pp. 995–1011.

---

## §5.2 ENO2 — Reconstruction essentiellement non oscillante

### Motivation

La méthode **ENO** (*Essentially Non-Oscillatory*), introduite par Harten, Engquist, Osher et Chakravarthy [Harten et al., 1987], propose une approche radicalement différente des limiteurs de pente : au lieu de limiter une pente fixe, on **choisit le stencil le plus lisse** parmi plusieurs candidats. L'idée est d'éviter d'interpoler à travers une discontinuité en sélectionnant automatiquement le stencil qui ne la traverse pas.

### Principe : sélection de stencil

Pour une reconstruction d'ordre 2 (ENO2), on dispose de deux stencils candidats pour calculer la pente dans la cellule $i$ :

- **Stencil gauche** : $\{i-1, i\}$ utilisant la différence $D_{i-1} = W_i - W_{i-1}$.
- **Stencil droit** : $\{i, i+1\}$ utilisant la différence $D_i = W_{i+1} - W_i$.

### Calcul des différences

Les **premières différences** (ou différences finies *forward*) sont :

$$D_i = \mathbf{W}_{i+1} - \mathbf{W}_i$$

Les **secondes différences** (différences divisées d'ordre 2) sont :

$$DD_i = D_{i+1} - D_i = \mathbf{W}_{i+2} - 2\mathbf{W}_{i+1} + \mathbf{W}_i$$

### Critère de sélection

Le stencil est choisi en comparant les secondes différences de part et d'autre :

$$\text{Si } |DD_{i-1}| \leq |DD_i| \implies \text{stencil gauche (pente } D_{i-1}\text{)}$$
$$\text{Sinon} \implies \text{stencil droit (pente } D_i\text{)}$$

L'idée est que la seconde différence la plus petite en valeur absolue correspond à la zone la plus lisse.

### États reconstruits

Pour l'état **gauche** à l'interface $i+1/2$ (reconstruction depuis la cellule $i$) :

$$\mathbf{W}_{i+1/2}^L = \mathbf{W}_i + \frac{1}{2} \, \text{slope}_L$$

où $\text{slope}_L$ est la pente sélectionnée par le critère ENO.

Pour l'état **droit** à l'interface $i+1/2$ (reconstruction depuis la cellule $i+1$) :

$$\mathbf{W}_{i+1/2}^R = \mathbf{W}_{i+1} - \frac{1}{2} \, \text{slope}_R$$

où $\text{slope}_R$ est la pente sélectionnée par le critère ENO appliqué à la cellule $i+1$.

### Avantages et inconvénients

**Avantages** :
- Pas de paramètre à régler (contrairement aux limiteurs qui offrent un choix entre plusieurs fonctions $\varphi$).
- Sélection automatique du stencil basée sur un critère objectif (régularité locale).
- Bonne capture des discontinuités sans oscillation.

**Inconvénients** :
- La **commutation abrupte** entre stencils peut introduire de petites perturbations quand la solution est proche du seuil de décision.
- Les poids ne sont pas des fonctions lisses de la solution (contrairement à WENO).
- Limité à l'ordre 2 dans cette implémentation.

### Cellules fantômes et ordre

- **Ordre spatial** : 2.
- **Cellules fantômes** : $n_{\text{ghost}} = 2$.
- **Intégrateur temporel par défaut** : RK2 (Heun).

### Références

- [Harten et al., 1987] A. Harten, B. Engquist, S. Osher, S. R. Chakravarthy, *Uniformly High Order Accurate Essentially Non-Oscillatory Schemes, III*, J. Comput. Phys., 71, pp. 231–303.

---

## §5.3 WENO3 — Weighted ENO d'ordre 3

### Motivation

La méthode ENO sélectionne un seul stencil et rejette les autres, ce qui constitue un gaspillage d'information. De plus, la commutation binaire entre stencils est non lisse. La méthode **WENO** (*Weighted ENO*), introduite par Liu, Osher et Chan (1994) puis formalisée par Jiang et Shu [Jiang, Shu, 1996], corrige ces deux défauts en **combinant tous les stencils** avec des poids non linéaires qui s'adaptent à la régularité locale.

En zone lisse, les poids tendent vers leurs valeurs **optimales** (qui maximisent l'ordre de précision). Près d'une discontinuité, le poids du stencil qui traverse la discontinuité tend vers zéro, reproduisant le comportement ENO.

### Deux sous-stencils

Pour WENO3, on utilise deux sous-stencils de 2 cellules chacun :

- $S_0 = \{i-1, i\}$ : stencil biaisé à gauche.
- $S_1 = \{i, i+1\}$ : stencil biaisé à droite.

Chaque stencil fournit une reconstruction linéaire de la solution.

### Indicateurs de régularité

Les **indicateurs de régularité** (*smoothness indicators*) mesurent la variation locale sur chaque sous-stencil. Pour WENO3, ils sont simplement le carré des premières différences :

$$\beta_0 = (D_{i-1})^2 = (\mathbf{W}_i - \mathbf{W}_{i-1})^2$$
$$\beta_1 = (D_i)^2 = (\mathbf{W}_{i+1} - \mathbf{W}_i)^2$$

Un indicateur $\beta_k$ élevé signifie que le stencil $S_k$ traverse une zone de forte variation (potentiellement une discontinuité).

### Poids idéaux

Les poids **idéaux** (ou *linear weights*) sont ceux qui, en zone parfaitement lisse, donnent l'ordre de précision maximal. Pour la reconstruction de l'état **gauche** :

$$d_0 = \frac{1}{3}, \qquad d_1 = \frac{2}{3}$$

Pour la reconstruction de l'état **droit** :

$$d_0 = \frac{2}{3}, \qquad d_1 = \frac{1}{3}$$

### Variante JS (Jiang-Shu)

Les poids non linéaires de Jiang-Shu [Jiang, Shu, 1996] sont :

$$\alpha_k = \frac{d_k}{(\varepsilon + \beta_k)^2}, \qquad \omega_k = \frac{\alpha_k}{\alpha_0 + \alpha_1}$$

avec $\varepsilon = 10^{-6}$, petit paramètre pour éviter la division par zéro.

**Comportement** :
- En zone lisse ($\beta_0 \approx \beta_1$), les poids convergent vers les poids idéaux $d_k$.
- Près d'une discontinuité ($\beta_k \gg \beta_l$), le poids $\omega_k \to 0$ : le stencil contaminé est exclu.

**Limitation** : aux **points critiques** (où les dérivées s'annulent, $\mathbf{W}' = 0$), les deux $\beta_k$ sont petits et du même ordre, mais les poids ne convergent pas assez vite vers les poids idéaux. L'ordre de précision est alors dégradé.

### Variante Z (WENO-Z, Borges et al.)

Pour corriger la perte de précision aux points critiques, Borges et al. [Borges et al., 2008] proposent un indicateur **global** :

$$\tau = |\beta_1 - \beta_0|$$

Les poids WENO-Z sont :

$$\alpha_k = d_k \left(1 + \left(\frac{\tau}{\beta_k + \varepsilon}\right)^2\right), \qquad \omega_k = \frac{\alpha_k}{\alpha_0 + \alpha_1}$$

avec $\varepsilon = \Delta x^2$ (et non une constante fixe). Ce choix de $\varepsilon$ proportionnel au maillage permet à l'indicateur $\tau / (\beta_k + \varepsilon)$ de rester bien conditionné pour toutes les tailles de maillage.

**Avantage** : aux points critiques, $\tau$ est d'ordre plus élevé que $\beta_k$, ce qui force les poids à converger vers les poids idéaux et **restaure l'ordre de précision optimal**.

### Reconstruction

L'état gauche à l'interface $i+1/2$ est :

$$\mathbf{W}_{i+1/2}^L = \mathbf{W}_i + \frac{1}{2} \bigl(\omega_0 \, D_{i-1} + \omega_1 \, D_i\bigr)$$

L'état droit se reconstruit de manière symétrique depuis la cellule $i+1$, avec les poids idéaux inversés.

### Cellules fantômes et ordre

- **Ordre spatial** : 3 (en zone lisse).
- **Cellules fantômes** : $n_{\text{ghost}} = 2$.
- **Intégrateur temporel par défaut** : SSP-RK3 (Shu-Osher), préservant la propriété TVD.

### Références

- [Jiang, Shu, 1996] G.-S. Jiang, C.-W. Shu, *Efficient Implementation of Weighted ENO Schemes*, J. Comput. Phys., 126, pp. 202–228.
- [Borges et al., 2008] R. Borges, M. Carmona, B. Costa, W. S. Don, *An Improved Weighted Essentially Non-Oscillatory Scheme for Hyperbolic Conservation Laws*, J. Comput. Phys., 227, pp. 3191–3211.

---

## §5.4 WENO5 — Weighted ENO d'ordre 5

### Motivation

WENO5 étend le principe de WENO3 en utilisant **trois sous-stencils** de 3 cellules chacun, atteignant l'ordre 5 en zone lisse. C'est le schéma WENO le plus utilisé en pratique pour les lois de conservation hyperboliques, offrant un excellent compromis entre précision élevée et robustesse près des discontinuités.

### Trois sous-stencils

Pour la reconstruction de l'état gauche à l'interface $i+1/2$, les trois sous-stencils sont :

- $S_0 = \{i, i+1, i+2\}$ : stencil biaisé à droite.
- $S_1 = \{i-1, i, i+1\}$ : stencil central.
- $S_2 = \{i-2, i-1, i\}$ : stencil biaisé à gauche.

### Indicateurs de régularité

Les indicateurs de régularité pour WENO5 sont plus complexes que pour WENO3, car ils intègrent les termes de dérivée seconde du polynôme d'interpolation :

$$\beta_0 = \frac{13}{12}(\mathbf{W}_i - 2\mathbf{W}_{i+1} + \mathbf{W}_{i+2})^2 + \frac{1}{4}(3\mathbf{W}_i - 4\mathbf{W}_{i+1} + \mathbf{W}_{i+2})^2$$

$$\beta_1 = \frac{13}{12}(\mathbf{W}_{i-1} - 2\mathbf{W}_i + \mathbf{W}_{i+1})^2 + \frac{1}{4}(\mathbf{W}_{i-1} - \mathbf{W}_{i+1})^2$$

$$\beta_2 = \frac{13}{12}(\mathbf{W}_{i-2} - 2\mathbf{W}_{i-1} + \mathbf{W}_i)^2 + \frac{1}{4}(\mathbf{W}_{i-2} - 4\mathbf{W}_{i-1} + 3\mathbf{W}_i)^2$$

Le premier terme de chaque $\beta_k$ est proportionnel au carré de la **dérivée seconde** approchée sur le stencil (terme dominant en $O(\Delta x^4)$), tandis que le second est proportionnel au carré de la **dérivée première** (terme en $O(\Delta x^2)$). Ensemble, ils mesurent la régularité totale de la solution sur le stencil.

### Poids idéaux

Pour la reconstruction de l'état **gauche** :

$$d_0 = \frac{3}{10}, \qquad d_1 = \frac{3}{5}, \qquad d_2 = \frac{1}{10}$$

Pour la reconstruction de l'état **droit**, les poids sont inversés :

$$d_0 = \frac{1}{10}, \qquad d_1 = \frac{3}{5}, \qquad d_2 = \frac{3}{10}$$

Le stencil central $S_1$ reçoit le poids le plus élevé ($3/5$) car il fournit la meilleure approximation centrée.

### Variante JS (Jiang-Shu)

$$\alpha_k = \frac{d_k}{(\varepsilon + \beta_k)^2}, \qquad \omega_k = \frac{\alpha_k}{\sum_{l=0}^{2} \alpha_l}$$

avec $\varepsilon = 10^{-6}$.

### Variante Z (WENO-Z)

L'indicateur global pour WENO5 est :

$$\tau_5 = |\beta_0 - \beta_2|$$

Les poids sont :

$$\alpha_k = d_k \left(1 + \left(\frac{\tau_5}{\beta_k + \varepsilon}\right)^2\right), \qquad \omega_k = \frac{\alpha_k}{\sum_{l=0}^{2} \alpha_l}$$

avec $\varepsilon = \Delta x^2$.

L'indicateur $\tau_5 = |\beta_0 - \beta_2|$ compare les stencils les plus éloignés, fournissant une mesure de la variation à grande échelle. En zone lisse, $\tau_5$ est d'ordre $O(\Delta x^5)$ alors que les $\beta_k$ sont d'ordre $O(\Delta x^2)$, ce qui permet aux poids de retrouver leurs valeurs optimales avec une erreur suffisamment petite pour préserver l'ordre 5.

### Reconstructions polynomiales

Chaque sous-stencil fournit un polynôme d'interpolation de degré 2. Pour l'état **gauche** à l'interface $i+1/2$ :

$$p_0 = \frac{1}{3}\mathbf{W}_i + \frac{5}{6}\mathbf{W}_{i+1} - \frac{1}{6}\mathbf{W}_{i+2}$$

$$p_1 = -\frac{1}{6}\mathbf{W}_{i-1} + \frac{5}{6}\mathbf{W}_i + \frac{1}{3}\mathbf{W}_{i+1}$$

$$p_2 = \frac{1}{3}\mathbf{W}_{i-2} - \frac{7}{6}\mathbf{W}_{i-1} + \frac{11}{6}\mathbf{W}_i$$

Pour l'état **droit** à l'interface $i+1/2$ (reconstruction depuis la cellule $i+1$, en notant $k = i+1$) :

$$p_0 = \frac{11}{6}\mathbf{W}_k - \frac{7}{6}\mathbf{W}_{k+1} + \frac{1}{3}\mathbf{W}_{k+2}$$

$$p_1 = \frac{1}{3}\mathbf{W}_{k-1} + \frac{5}{6}\mathbf{W}_k - \frac{1}{6}\mathbf{W}_{k+1}$$

$$p_2 = -\frac{1}{6}\mathbf{W}_{k-2} + \frac{5}{6}\mathbf{W}_{k-1} + \frac{1}{3}\mathbf{W}_k$$

La valeur reconstruite finale est la combinaison pondérée :

$$\mathbf{W}^{L,R}_{i+1/2} = \sum_{k=0}^{2} \omega_k \, p_k$$

### Comparaison JS vs Z aux points critiques

| Propriété | JS | Z |
|---|---|---|
| Précision en zone lisse | Ordre 5 | Ordre 5 |
| Précision aux points critiques ($\mathbf{W}' = 0$) | **Dégradée** (ordre 3-4) | **Restaurée** (ordre 5) |
| Paramètre $\varepsilon$ | $10^{-6}$ (constant) | $\Delta x^2$ (adaptatif) |
| Coût supplémentaire | — | Calcul de $\tau_5$ (négligeable) |

En pratique, la variante Z est recommandée pour les solutions contenant des extrema lisses (ondes acoustiques, tourbillons) car elle préserve l'ordre de convergence théorique.

### Cellules fantômes et ordre

- **Ordre spatial** : 5 (en zone lisse).
- **Cellules fantômes** : $n_{\text{ghost}} = 3$ (stencils de 5 cellules).
- **Intégrateur temporel par défaut** : Dormand-Prince RK5 (6 étages), pour que l'erreur temporelle n'entame pas la précision spatiale d'ordre 5.

### Références

- [Jiang, Shu, 1996] G.-S. Jiang, C.-W. Shu, *Efficient Implementation of Weighted ENO Schemes*, J. Comput. Phys., 126, pp. 202–228.
- [Borges et al., 2008] R. Borges, M. Carmona, B. Costa, W. S. Don, *An Improved Weighted Essentially Non-Oscillatory Scheme for Hyperbolic Conservation Laws*, J. Comput. Phys., 227, pp. 3191–3211.

---

## §5.5 Tableau comparatif des reconstructions

| Propriété | MUSCL | ENO2 | WENO3 | WENO5 |
|---|---|---|---|---|
| **Ordre spatial** | 2 | 2 | 3 | 5 |
| **Cellules fantômes** ($n_{\text{ghost}}$) | 2 | 2 | 2 | 3 |
| **Nombre de stencils** | 1 (pente limitée) | 2 (sélection) | 2 (pondération) | 3 (pondération) |
| **Contrôle des oscillations** | Limiteur TVD | Sélection de stencil | Poids non linéaires | Poids non linéaires |
| **Précision aux points critiques (JS)** | — | — | Dégradée | Dégradée |
| **Précision aux points critiques (Z)** | — | — | **Restaurée** | **Restaurée** |
| **Coût relatif** | Faible | Faible | Modéré | Élevé |
| **Intégrateur par défaut** | RK2 | RK2 | SSP-RK3 | RK5 (Dormand-Prince) |
| **Paramètres utilisateur** | Choix du limiteur | Aucun | Variante JS/Z | Variante JS/Z |
| **Composable avec** | Rusanov, HLL, HLLC, Roe, Godunov | idem | idem | idem |

**Recommandations pratiques** :

- Pour un **premier calcul rapide** ou un problème dominé par des chocs forts : **MUSCL + van Leer** offre le meilleur compromis robustesse/coût.
- Pour une **étude de convergence** en maillage ou une solution lisse : **WENO5-Z** préserve l'ordre 5 même aux extrema.
- Pour une **capture fine des contacts** : **MUSCL + superbee** (avec prudence sur les profils lisses) ou **WENO5**.
- Le choix du **flux numérique** (HLLC recommandé) a souvent plus d'impact que le choix de la reconstruction sur la qualité de la discontinuité de contact.
