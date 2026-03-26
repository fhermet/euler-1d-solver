# Chapitre 2 : Problème de Riemann

## Introduction

Le **problème de Riemann** est un problème de Cauchy pour un système hyperbolique de lois de conservation dont les données initiales sont **constantes par morceaux**, avec une unique discontinuité en $x = x_0$ :

$$
\mathbf{U}(x, 0) = \begin{cases} \mathbf{U}_L & \text{si } x < x_0 \\ \mathbf{U}_R & \text{si } x > x_0 \end{cases}
$$

où $\mathbf{U}_L = (\rho_L, \rho_L u_L, E_L)^T$ et $\mathbf{U}_R = (\rho_R, \rho_R u_R, E_R)^T$ sont deux états constants donnés, séparés par une discontinuité initiale en $x_0$.

Ce problème, apparemment simple, est la **pierre angulaire** de la mécanique des fluides numérique pour plusieurs raisons :

1. **Fondement des méthodes de volumes finis.** Dans un schéma de Godunov, chaque interface entre deux cellules voisines constitue un problème de Riemann local. Les états gauche et droit sont les valeurs moyennes dans les cellules adjacentes. Le flux numérique à l'interface est déterminé par la solution du problème de Riemann (cf. Chapitre 3).

2. **Solution de référence.** Le problème de Riemann admet une solution exacte analytique. Cette solution sert de référence pour valider les schémas numériques et mesurer leurs erreurs de dissipation et de dispersion.

3. **Contenu physique maximal.** Malgré la simplicité des données initiales, la solution contient les trois types fondamentaux d'ondes des équations d'Euler : ondes de choc, détentes et discontinuités de contact.

Ce chapitre développe la théorie complète du problème de Riemann pour les équations d'Euler 1D d'un gaz parfait, depuis la structure de la solution jusqu'à l'algorithme de résolution numérique. L'implémentation se trouve dans le module [`euler1d/riemann.py`](../euler1d/riemann.py).

---

## §2.1 Structure de la solution

### Auto-similarité

La donnée initiale ne possède aucune échelle de longueur ni de temps intrinsèque : elle est entièrement caractérisée par les deux états constants $\mathbf{U}_L$ et $\mathbf{U}_R$ et la position $x_0$. Par conséquent, la solution ne dépend pas de $x$ et $t$ indépendamment, mais uniquement de la **variable de similarité** :

$$
\xi = \frac{x - x_0}{t}
$$

On dit que la solution est **auto-similaire** : la structure des ondes à $t = 0.1$ est une version dilatée (par un facteur 10) de celle à $t = 0.01$. Toutes les ondes sont des droites dans le plan $(x, t)$ issues du point $(x_0, 0)$.

### Trois ondes, quatre états

Comme démontré au §1.5 du Chapitre 1, le système d'Euler 1D possède trois valeurs propres $\lambda_1 = u - a$, $\lambda_2 = u$ et $\lambda_3 = u + a$. La solution du problème de Riemann est donc composée de **trois ondes** séparant **quatre états constants** :

$$
\mathbf{U}_L \xrightarrow{\text{onde 1}} \mathbf{U}_L^\ast \xrightarrow{\text{onde 2}} \mathbf{U}_R^\ast \xrightarrow{\text{onde 3}} \mathbf{U}_R
$$

```
         t
         ^
         |     /onde 1        | onde 2       \onde 3
         |    /               |               \
         |   /                |                \
         |  /     U*_L        |     U*_R        \
         | /                  |                  \
         |/                   |                   \
    U_L  |                    |                    |  U_R
   ------+---------------------------------------------> x
                             x_0
```

Les **conditions de compatibilité** à travers la discontinuité de contact (onde 2) imposent que la pression et la vitesse soient continues :

$$
p_L^\ast = p_R^\ast = p^\ast, \qquad u_L^\ast = u_R^\ast = u^\ast
$$

Les deux états intermédiaires ne diffèrent que par leur **densité** : $\rho_L^\ast \neq \rho_R^\ast$ en général.

La résolution du problème de Riemann se ramène donc à déterminer deux inconnues scalaires $(p^\ast, u^\ast)$, appelées **pression et vitesse de la région étoile**. Une fois ces quantités connues, les densités $\rho_L^\ast$ et $\rho_R^\ast$ s'en déduisent, ainsi que la nature (choc ou détente) de chaque onde acoustique.

---

## §2.2 Ondes de choc

### Relations de Rankine-Hugoniot

Un choc est une discontinuité qui se propage à une vitesse $s$. La solution étant discontinue, les équations d'Euler sous forme différentielle n'ont plus de sens. Il faut revenir à la **forme intégrale** des lois de conservation, qui fournit les **relations de Rankine-Hugoniot** :

$$
\mathbf{F}(\mathbf{U}_R) - \mathbf{F}(\mathbf{U}_L) = s \left( \mathbf{U}_R - \mathbf{U}_L \right)
$$

ou de manière abrégée :

$$
{[}\!{[} \mathbf{F} {]}\!{]} = s \, {[}\!{[} \mathbf{U} {]}\!{]}
$$

où ${[}\!{[} Q {]}\!{]} = Q_{\text{après}} - Q_{\text{avant}}$ désigne le saut de la quantité $Q$ à travers le choc, et $s$ est la **vitesse du choc**.

**Interprétation physique.** Les relations de Rankine-Hugoniot expriment la conservation de la masse, de la quantité de mouvement et de l'énergie à travers la discontinuité. Le flux net sortant du choc (membre de gauche) est exactement compensé par la variation des quantités conservatives due au déplacement du choc (membre de droite).

### Vitesse du choc

En développant les relations de Rankine-Hugoniot pour l'équation de conservation de la masse, on obtient la vitesse du choc. Pour un choc gauche (onde 1) reliant $\mathbf{U}_L$ à $\mathbf{U}_L^\ast$ :

$$
s_L = u_L - a_L \sqrt{\frac{\gamma + 1}{2\gamma} \frac{p^\ast}{p_L} + \frac{\gamma - 1}{2\gamma}}
$$

La quantité sous la racine est toujours supérieure à 1 quand $p^\ast > p_L$ (compression), de sorte que $\lvert s_L - u_L \rvert > a_L$ : **le choc se propage plus vite que le son** dans le milieu non perturbé.

Pour un choc droit (onde 3) reliant $\mathbf{U}_R^\ast$ à $\mathbf{U}_R$ :

$$
s_R = u_R + a_R \sqrt{\frac{\gamma + 1}{2\gamma} \frac{p^\ast}{p_R} + \frac{\gamma - 1}{2\gamma}}
$$

### Lieu de Hugoniot

L'ensemble des états $(\rho, u, p)$ accessibles depuis un état donné $(\rho_K, u_K, p_K)$ par un choc vérifie les relations de Rankine-Hugoniot. Cet ensemble forme le **lieu de Hugoniot** (ou courbe de Hugoniot) dans l'espace des états.

Pour un choc de la $k$-ème famille, la relation entre la pression dans la région étoile $p^\ast$ et la variation de vitesse s'écrit :

$$
f_K^{\text{choc}}(p^\ast) = (p^\ast - p_K) \sqrt{\frac{A_K}{p^\ast + B_K}}
$$

avec les coefficients :

$$
A_K = \frac{2}{(\gamma + 1) \rho_K}, \qquad B_K = \frac{\gamma - 1}{\gamma + 1} p_K
$$

**Origine de $A_K$.** Le coefficient $A_K = 2 / ((\gamma + 1) \rho_K)$ provient des relations de Rankine-Hugoniot appliquées au gaz parfait. Le facteur $(\gamma + 1)$ reflète le rapport de compression maximal qu'un choc peut produire dans un gaz parfait, qui est $(\gamma + 1) / (\gamma - 1)$.

**Origine de $B_K$.** Le coefficient $B_K = (\gamma - 1) / (\gamma + 1) \cdot p_K$ est lié à la pression de référence. Il assure que lorsque $p^\ast = p_K$ (pas de choc), la fonction $f_K^{\text{choc}}$ s'annule. Le rapport $(\gamma - 1) / (\gamma + 1)$ apparaît naturellement dans les relations de saut pour un gaz parfait.

### Condition d'entropie de Lax

Les relations de Rankine-Hugoniot admettent en général plusieurs solutions (chocs, discontinuités de contact, chocs non physiques). Pour sélectionner la solution **physiquement admissible**, on impose la **condition d'entropie de Lax** :

$$
\lambda_k(\mathbf{U}_{\text{avant}}) > s > \lambda_k(\mathbf{U}_{\text{après}})
$$

**Interprétation physique.** Les caractéristiques de la famille $k$ doivent **converger vers le choc** des deux côtés : les caractéristiques amont rattrapent le choc, et les caractéristiques aval sont rattrapées par le choc. Un choc est une compression : l'information « entre » dans le choc et ne peut pas en « sortir ». Un choc de détente (expansion), bien que mathématiquement possible, violerait cette condition et le second principe de la thermodynamique (l'entropie diminuerait).

---

## §2.3 Ondes de détente (raréfaction)

### Invariants de Riemann

Contrairement aux chocs, les détentes sont des ondes **continues et lisses**. Elles apparaissent lorsque le fluide se détend ($p^\ast < p_K$). À travers une détente, l'écoulement est **isentropique** (entropie constante).

Les **invariants de Riemann** sont des quantités qui restent constantes le long des courbes caractéristiques d'une famille donnée :

- **1er champ** ($\lambda_1 = u - a$) : l'invariant $u + \frac{2a}{\gamma - 1}$ est constant le long des caractéristiques de la famille 1 (celles qui traversent la détente gauche sans y appartenir).
- **3e champ** ($\lambda_3 = u + a$) : l'invariant $u - \frac{2a}{\gamma - 1}$ est constant le long des caractéristiques de la famille 3.

**Interprétation physique.** L'invariant de Riemann relie la vitesse $u$ et la vitesse du son $a$ (donc la pression, via l'isentropie). Il exprime une **relation de compatibilité** entre les variations de vitesse et de pression à travers l'onde : lorsque la pression diminue (détente), la vitesse augmente, et vice versa. Le facteur $2/(\gamma - 1)$ quantifie le « taux de change » entre vitesse et vitesse du son, et dépend de la rigidité du gaz.

### Structure en éventail

La détente est un **éventail de caractéristiques** (fan) centré au point $(x_0, 0)$. Chaque caractéristique du fan porte une valeur constante de la variable de similarité $\xi = x/t$.

Pour une détente gauche (1er champ), le fan est délimité par :
- **Tête** (bord extérieur) : $\xi_{\text{head}} = u_L - a_L$ (caractéristique la plus rapide, dans l'état non perturbé)
- **Queue** (bord intérieur) : $\xi_{\text{tail}} = u^\ast - a_L^\ast$ (caractéristique dans la région étoile)

Pour une détente droite (3e champ) :
- **Tête** : $\xi_{\text{head}} = u_R + a_R$
- **Queue** : $\xi_{\text{tail}} = u^\ast + a_R^\ast$

### Profils auto-similaires à l'intérieur du fan

En utilisant l'invariant de Riemann et la relation isentropique, on obtient les profils exacts à l'intérieur du fan gauche. Pour $\xi_{\text{head}} \leq \xi \leq \xi_{\text{tail}}$ :

**Vitesse dans le fan :**

$$
u_{\text{fan}} = \frac{2}{\gamma + 1}\left(a_L + \frac{\gamma - 1}{2} u_L + \xi\right)
$$

Cette formule provient de l'invariant de Riemann $u + 2a/(\gamma - 1) = u_L + 2a_L/(\gamma - 1)$, combiné avec la condition $\xi = u - a$ (la vitesse de similarité est la vitesse caractéristique locale). Les deux équations donnent un système linéaire en $(u, a)$ dont la solution est la formule ci-dessus.

**Vitesse du son dans le fan :**

$$
a_{\text{fan}} = a_L - \frac{\gamma - 1}{2}(u_{\text{fan}} - u_L)
$$

C'est l'invariant de Riemann réécrit : toute augmentation de vitesse $u$ se fait au détriment de la vitesse du son $a$ (et donc de la pression et de la densité). Le facteur $(\gamma - 1)/2$ est le coefficient de conversion.

**Densité dans le fan :**

$$
\rho_{\text{fan}} = \rho_L \left(\frac{a_{\text{fan}}}{a_L}\right)^{2/(\gamma - 1)}
$$

Cette relation provient de l'isentropie : pour un processus isentropique dans un gaz parfait, $\rho \propto a^{2/(\gamma - 1)}$, car $a^2 = \gamma p / \rho$ et $p / \rho^\gamma = \text{const}$ le long d'une isentrope.

**Pression dans le fan :**

$$
p_{\text{fan}} = p_L \left(\frac{a_{\text{fan}}}{a_L}\right)^{2\gamma/(\gamma - 1)}
$$

De même, pour un processus isentropique, $p \propto a^{2\gamma/(\gamma - 1)}$. L'exposant $2\gamma/(\gamma - 1)$ est plus grand que $2/(\gamma - 1)$ (celui de la densité), ce qui signifie que la pression varie plus rapidement que la densité à travers la détente.

### Branche raréfaction de l'équation de pression

La relation entre la pression de la région étoile $p^\ast$ et la variation de vitesse à travers une détente s'écrit :

$$
f_K^{\text{rare}}(p^\ast) = \frac{2 a_K}{\gamma - 1}\left[\left(\frac{p^\ast}{p_K}\right)^{(\gamma - 1)/(2\gamma)} - 1\right]
$$

**Origine.** Cette expression s'obtient en intégrant l'invariant de Riemann entre l'état $K$ et la région étoile. Le terme $(p^\ast/p_K)^{(\gamma-1)/(2\gamma)}$ est le rapport des vitesses du son $a^\ast/a_K$, obtenu via la relation isentropique $a \propto p^{(\gamma-1)/(2\gamma)}$. Lorsque $p^\ast < p_K$ (détente), ce rapport est inférieur à 1 et $f_K^{\text{rare}}$ est négatif : la vitesse change dans le sens de la détente.

---

## §2.4 Discontinuité de contact

### Champ linéairement dégénéré

La deuxième valeur propre $\lambda_2 = u$ est associée à un **champ linéairement dégénéré** (cf. §1.5) :

$$
\nabla_{\mathbf{U}} \lambda_2 \cdot \mathbf{r}_2 = 0
$$

Contrairement aux champs véritablement non linéaires qui produisent des chocs ou des détentes, un champ linéairement dégénéré produit toujours le même type d'onde : une **discontinuité de contact**.

### Propriétés de la discontinuité de contact

La discontinuité de contact se propage à la vitesse $u^\ast$ et sépare les régions étoile gauche et droite. À travers cette discontinuité :

| Grandeur | Comportement | Explication |
|---|---|---|
| Pression $p$ | **Continue** : $p_L^\ast = p_R^\ast = p^\ast$ | L'équilibre des forces à travers l'interface impose la continuité de la pression. Un saut de pression créerait une accélération infinie. |
| Vitesse $u$ | **Continue** : $u_L^\ast = u_R^\ast = u^\ast$ | Les deux masses de fluide se déplacent à la même vitesse. Un saut de vitesse impliquerait une séparation ou une interpénétration du fluide. |
| Densité $\rho$ | **Discontinue** : $\rho_L^\ast \neq \rho_R^\ast$ | Deux masses de fluide de densités différentes peuvent coexister à la même pression et vitesse. |
| Température $T$ | **Discontinue** | Par l'équation d'état $p = \rho R T / M$, si $p$ est continu et $\rho$ discontinu, alors $T$ l'est aussi. |
| Entropie $s$ | **Discontinue** | Le saut d'entropie est lié au saut de densité à pression constante. Mais il n'y a **pas de production d'entropie** : le saut est simplement transporté. |

### Exemple physique

Un exemple concret de discontinuité de contact est l'interface entre deux gaz différents (par exemple, de l'hélium et de l'air) dans un tube. Les deux gaz sont à la même pression et se déplacent à la même vitesse, mais ils ont des densités (et des masses molaires) très différentes. L'interface est advectée à la vitesse du fluide sans se déformer.

Dans le problème de Riemann, la discontinuité de contact est créée même si les deux états initiaux sont le même gaz (même $\gamma$). Elle provient du fait que les ondes acoustiques (choc ou détente) de part et d'autre compriment ou détendent le gaz de manière différente, produisant deux densités distinctes $\rho_L^\ast$ et $\rho_R^\ast$ dans la région étoile.

---

## §2.5 Région étoile et équation de pression

### Formulation du problème

Les conditions de compatibilité à travers la discontinuité de contact imposent $p_L^\ast = p_R^\ast = p^\ast$ et $u_L^\ast = u_R^\ast = u^\ast$. La vitesse dans la région étoile s'exprime en fonction de $p^\ast$ via les relations de chaque onde :

$$
u^\ast = u_L - f_L(p^\ast) = u_R + f_R(p^\ast)
$$

où $f_K(p^\ast)$ est la variation de vitesse à travers l'onde de la famille $K$ :
- si $p^\ast > p_K$ (choc) : $f_K = f_K^{\text{choc}}$ (§2.2),
- si $p^\ast \leq p_K$ (détente) : $f_K = f_K^{\text{rare}}$ (§2.3).

En éliminant $u^\ast$, on obtient l'**équation de pression** — une unique équation scalaire non linéaire en $p^\ast$ :

$$
f(p^\ast) = f_L(p^\ast) + f_R(p^\ast) + (u_R - u_L) = 0
$$

### Branche choc

Si $p^\ast > p_K$ (l'onde $K$ est un choc), la contribution de ce côté à l'équation de pression est :

$$
f_K^{\text{choc}}(p^\ast) = (p^\ast - p_K) \sqrt{\frac{A_K}{p^\ast + B_K}}
$$

avec :

$$
A_K = \frac{2}{(\gamma + 1) \rho_K}, \qquad B_K = \frac{\gamma - 1}{\gamma + 1} p_K
$$

**Dérivation.** Cette formule s'obtient en combinant les trois relations de Rankine-Hugoniot pour le système d'Euler. En éliminant la vitesse du choc $s$ et la densité post-choc entre les trois équations, on isole la variation de vitesse en fonction de la pression post-choc. Les coefficients $A_K$ et $B_K$ résultent de l'algèbre spécifique au gaz parfait (relation de Hugoniot adiabatique).

La fonction $f_K^{\text{choc}}$ est positive et croissante pour $p^\ast > p_K$ : plus la compression est forte, plus la variation de vitesse est grande.

### Branche détente

Si $p^\ast \leq p_K$ (l'onde $K$ est une détente), la contribution est :

$$
f_K^{\text{rare}}(p^\ast) = \frac{2 a_K}{\gamma - 1}\left[\left(\frac{p^\ast}{p_K}\right)^{(\gamma - 1)/(2\gamma)} - 1\right]
$$

**Dérivation.** On part de l'invariant de Riemann : $u^\ast = u_K \mp \frac{2}{\gamma - 1}(a^\ast - a_K)$ (signe $-$ pour le côté gauche, $+$ pour le droit). La vitesse du son $a^\ast$ dans la région étoile est reliée à la pression par la relation isentropique :

$$
\frac{a^\ast}{a_K} = \left(\frac{p^\ast}{p_K}\right)^{(\gamma - 1)/(2\gamma)}
$$

car pour un processus isentropique dans un gaz parfait, $p / \rho^\gamma = \text{const}$ et $a = \sqrt{\gamma p / \rho}$. En substituant, on obtient la formule ci-dessus.

La fonction $f_K^{\text{rare}}$ est négative pour $p^\ast < p_K$ et s'annule en $p^\ast = p_K$.

### Continuité des branches

Les deux branches se raccordent continûment en $p^\ast = p_K$ :

$$
f_K^{\text{choc}}(p_K) = 0 = f_K^{\text{rare}}(p_K)
$$

Les dérivées premières se raccordent également, garantissant que $f_K$ est de classe $C^1$. La fonction totale $f(p^\ast)$ est donc $C^1$ sur $]0, +\infty[$, ce qui assure la bonne convergence de la méthode de Newton.

---

## §2.6 Résolution par itération de Newton

### Estimation initiale PVRS

L'itération de Newton nécessite une estimation initiale $p^{(0)}$ suffisamment proche de la solution pour garantir la convergence. L'estimation **PVRS** (*Primitive Variable Riemann Solver*) est obtenue en linéarisant les équations d'Euler autour d'un état moyen :

$$
p_{\text{PV}} = \frac{1}{2}(p_L + p_R) - \frac{1}{2}(u_R - u_L) \cdot \frac{\rho_L a_L + \rho_R a_R}{\rho_L + \rho_R}
$$

**Interprétation physique.** Le premier terme $\frac{1}{2}(p_L + p_R)$ est la moyenne arithmétique des pressions. Le second terme est une **correction acoustique** : si les fluides s'éloignent ($u_R - u_L > 0$, divergence), la pression diminue (détente) ; s'ils convergent ($u_R - u_L < 0$), la pression augmente (compression). L'impédance acoustique $\rho a$ pondère cette correction. Le dénominateur $\rho_L + \rho_R$ normalise par la masse totale.

L'estimation est tronquée à une valeur positive minimale ($10^{-14}$) pour éviter les pressions négatives non physiques.

### Itération de Newton-Raphson

On résout $f(p^\ast) = 0$ par la méthode de Newton :

$$
p^{(k+1)} = p^{(k)} - \frac{f(p^{(k)})}{f'(p^{(k)})}
$$

La dérivée $f'(p) = f_L'(p) + f_R'(p)$ se calcule analytiquement pour chaque branche :

**Dérivée de la branche choc** ($p > p_K$) :

$$
f_K^{\text{choc}\prime}(p) = \sqrt{\frac{A_K}{p + B_K}} \left(1 - \frac{p - p_K}{2(p + B_K)}\right)
$$

Cette expression s'obtient par dérivation directe de $f_K^{\text{choc}}$ par rapport à $p$, en utilisant la règle du produit sur $(p - p_K) \times (A_K / (p + B_K))^{1/2}$.

**Dérivée de la branche détente** ($p \leq p_K$) :

$$
f_K^{\text{rare}\prime}(p) = \frac{1}{\rho_K a_K} \left(\frac{p}{p_K}\right)^{-(\gamma + 1)/(2\gamma)}
$$

Elle s'obtient en dérivant $(p/p_K)^{(\gamma-1)/(2\gamma)}$ par rapport à $p$, ce qui donne un exposant $(\gamma-1)/(2\gamma) - 1 = -(\gamma+1)/(2\gamma)$.

### Convergence

Le critère d'arrêt est un critère **relatif** :

$$
\frac{\lvert p^{(k+1)} - p^{(k)} \rvert}{\max(1, \lvert p^{(k+1)} \rvert)} < \varepsilon
$$

avec $\varepsilon = 10^{-10}$ par défaut. La convergence est **quadratique** (typique de Newton) : le nombre de décimales exactes double à chaque itération. En pratique, 3 à 6 itérations suffisent.

À chaque itération, on impose $p^{(k+1)} \geq 10^{-14}$ pour rester dans le domaine physique des pressions positives.

### Détection du vide

Avant de lancer l'itération, on vérifie si la solution admet une pression positive. Le **vide** se forme lorsque les deux fluides s'éloignent trop vite pour que les détentes puissent rester connectées. Le critère est :

$$
(u_R - u_L) \geq \frac{2}{\gamma - 1}(a_L + a_R)
$$

**Interprétation physique.** Le membre de gauche est la vitesse de séparation des deux fluides. Le membre de droite $\frac{2}{\gamma - 1}(a_L + a_R)$ est la **vitesse maximale de détente** : c'est la vitesse maximale que les détentes gauche et droite peuvent communiquer au fluide. Si la séparation excède cette vitesse maximale, une zone de vide (pression nulle) se forme au centre.

Lorsque cette condition est détectée, le solveur lève une erreur, car le modèle du gaz parfait ne décrit pas le vide.

---

## §2.7 Échantillonnage de la solution

### Principe

Une fois les quantités de la région étoile $(p^\ast, u^\ast, \rho_L^\ast, \rho_R^\ast)$ calculées, il faut déterminer l'état $(\rho, u, p)$ en tout point $\xi = (x - x_0) / t$. C'est l'opération d'**échantillonnage** (*sampling*).

### Densités dans la région étoile

Les densités $\rho_L^\ast$ et $\rho_R^\ast$ se calculent différemment selon la nature de l'onde :

**À travers un choc** ($p^\ast > p_K$) — relation de Rankine-Hugoniot :

$$
\rho_K^\ast = \rho_K \frac{\dfrac{p^\ast}{p_K} + \dfrac{\gamma - 1}{\gamma + 1}}{\dfrac{\gamma - 1}{\gamma + 1} \dfrac{p^\ast}{p_K} + 1}
$$

Cette formule provient de l'élimination de la vitesse du choc dans les relations de Rankine-Hugoniot. Elle montre que pour un choc infiniment fort ($p^\ast/p_K \to \infty$), la densité tend vers $\rho_K \cdot (\gamma + 1) / (\gamma - 1)$, qui est le **taux de compression maximal** d'un choc dans un gaz parfait ($= 6$ pour $\gamma = 1.4$).

**À travers une détente** ($p^\ast \leq p_K$) — relation isentropique :

$$
\rho_K^\ast = \rho_K \left(\frac{p^\ast}{p_K}\right)^{1/\gamma}
$$

C'est la relation de Poisson $p / \rho^\gamma = \text{const}$ réécrite pour exprimer $\rho^\ast$ en fonction de $p^\ast$.

### Arbre de décision

L'échantillonnage suit un **arbre de décision** basé sur la position $\xi = (x - x_0)/t$ par rapport aux différentes ondes :

```
                        ξ < u* ?
                       /        \
                    oui          non
                   /                \
             CÔTÉ GAUCHE        CÔTÉ DROIT
               /                      \
          p* > p_L ?              p* > p_R ?
          /       \               /       \
        oui      non            oui      non
       CHOC    DÉTENTE         CHOC    DÉTENTE
       /  \     /  |  \        /  \     /  |  \
   ξ<s_L  ξ≥s_L  ξ<ξ_h ξ∈fan ξ>ξ_t  ξ≤s_R  ξ>s_R  ξ≤ξ_h ξ∈fan ξ>ξ_t
    U_L   U*_L   U_L  fan  U*_L    U*_R    U_R   U*_R  fan   U_R
```

Détaillons chaque cas :

**1. $\xi < u^\ast$ (côté gauche de la discontinuité de contact)**

- **Choc gauche** ($p^\ast > p_L$) :
  - Si $\xi < s_L$ : état non perturbé $(\rho_L, u_L, p_L)$
  - Si $\xi \geq s_L$ : région étoile gauche $(\rho_L^\ast, u^\ast, p^\ast)$

- **Détente gauche** ($p^\ast \leq p_L$) :
  - Si $\xi < u_L - a_L$ (avant la tête) : état non perturbé $(\rho_L, u_L, p_L)$
  - Si $u_L - a_L \leq \xi \leq u^\ast - a_L^\ast$ (dans le fan) : profils auto-similaires du §2.3
  - Si $\xi > u^\ast - a_L^\ast$ (après la queue) : région étoile gauche $(\rho_L^\ast, u^\ast, p^\ast)$

**2. $\xi \geq u^\ast$ (côté droit de la discontinuité de contact)**

- **Choc droit** ($p^\ast > p_R$) :
  - Si $\xi > s_R$ : état non perturbé $(\rho_R, u_R, p_R)$
  - Si $\xi \leq s_R$ : région étoile droite $(\rho_R^\ast, u^\ast, p^\ast)$

- **Détente droite** ($p^\ast \leq p_R$) :
  - Si $\xi > u_R + a_R$ (avant la tête) : état non perturbé $(\rho_R, u_R, p_R)$
  - Si $u^\ast + a_R^\ast \leq \xi \leq u_R + a_R$ (dans le fan) : profils auto-similaires (symétriques du §2.3)
  - Si $\xi < u^\ast + a_R^\ast$ (après la queue) : région étoile droite $(\rho_R^\ast, u^\ast, p^\ast)$

### Vectorisation

Dans l'implémentation, l'échantillonnage est **vectorisé** : au lieu de boucler en Python sur chaque position $x_i$, on utilise des **masques booléens** NumPy pour appliquer les formules à des tableaux entiers en une seule opération. Cela accélère considérablement le calcul lorsqu'on évalue la solution sur un maillage fin (typiquement $N = 100$ à $10\,000$ cellules).

---

## §2.8 Avantages et inconvénients du solveur exact

### Avantages

1. **Solution de référence.** Le solveur exact fournit la solution analytique du problème de Riemann, sans aucune approximation. C'est la référence absolue pour valider les schémas numériques et quantifier leurs erreurs (dissipation, dispersion, oscillations).

2. **Schéma de Godunov.** Le solveur exact peut être utilisé comme flux numérique dans le schéma de Godunov (cf. Chapitre 3). Ce schéma résout un problème de Riemann exact à chaque interface à chaque pas de temps. C'est le schéma de volumes finis « idéal » au sens où il capture parfaitement la physique des ondes aux interfaces.

3. **Robustesse.** L'itération de Newton converge de manière fiable pour tous les cas physiquement admissibles (pressions et densités positives, pas de vide). La détection du vide évite les divergences.

### Inconvénients

1. **Coût calculatoire.** Chaque problème de Riemann nécessite une itération de Newton (3 à 6 itérations, chacune impliquant l'évaluation de fonctions non linéaires). Pour un maillage de $N$ cellules, il faut résoudre $N + 1$ problèmes de Riemann à chaque pas de temps. Le coût total est significativement plus élevé que celui des solveurs approchés.

2. **Complexité d'implémentation.** L'arbre de décision de l'échantillonnage (§2.7) comporte de nombreux cas (choc/détente, gauche/droite, dans/hors du fan). Cela rend le code plus complexe et plus sujet aux erreurs que les solveurs approchés.

3. **Limité au gaz parfait.** La résolution exacte repose sur les formules analytiques spécifiques au gaz parfait ($p = (\gamma - 1)\rho e$). Pour des équations d'état plus générales (gaz réels, mélanges réactifs), le solveur exact est beaucoup plus difficile à formuler.

### Solveurs approchés

Pour pallier le coût du solveur exact, des **solveurs de Riemann approchés** ont été développés. Ils fournissent une estimation du flux numérique sans itération, en exploitant des simplifications de la structure des ondes :

- **Rusanov** (Lax-Friedrichs local) : une seule onde de vitesse maximale.
- **HLL** (Harten-Lax-van Leer) : deux ondes, pas de discontinuité de contact.
- **HLLC** (HLL-Contact) : trois ondes, restaure la discontinuité de contact.
- **Roe** : linéarisation de la jacobienne autour d'un état moyen.

Ces solveurs seront présentés en détail au Chapitre 4.

---

## §2.9 Implémentation dans le code

Les concepts mathématiques de ce chapitre sont implémentés dans le module [`euler1d/riemann.py`](../euler1d/riemann.py). Le tableau suivant établit la correspondance entre les formules et les fonctions :

| Concept | Section | Fonction |
|---|---|---|
| Branche choc de $f_K$ | §2.2, §2.5 | `_f_shock()` |
| Branche détente de $f_K$ | §2.3, §2.5 | `_f_rare()` |
| Sélection de branche | §2.5 | `_f_side()` |
| Équation de pression $f(p^\ast)$ | §2.5 | `_f_total()` |
| Dérivée $f'(p^\ast)$ | §2.6 | `_df_total()`, `_df_dp_side()` |
| Estimation initiale PVRS | §2.6 | `_pressure_guess_pvrs()` |
| Détection du vide | §2.6 | `_vacuum_will_form()` |
| Itération de Newton | §2.6 | `solve_p_star()` |
| Vitesse $u^\ast$ | §2.5 | `compute_u_star()` |
| Densités $\rho_L^\ast$, $\rho_R^\ast$ | §2.7 | `star_region_densities()` |
| Résolution complète | — | `solve_riemann()` |
| Échantillonnage vectorisé | §2.7 | `sample_riemann()` |
| Échantillonnage à $\xi = 0$ (Godunov) | §2.7 | `sample_at_interface()` |

Les cas test du problème de Riemann (Sod, Lax, double détente) sont définis dans [`euler1d/test_cases.py`](../euler1d/test_cases.py).

---

## Références

- **[Toro, 2009]** E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e édition, Springer, 2009. Chapitres 4-5 : solution exacte du problème de Riemann, itération de Newton, échantillonnage. La référence principale pour ce chapitre.

- **[Godunov, 1959]** S.K. Godunov, *A difference method for the numerical calculation of discontinuous solutions of the equations of hydrodynamics*, Matematicheskii Sbornik, 47(3):271-306, 1959. Article fondateur : première utilisation du problème de Riemann comme brique de base d'un schéma numérique.

- **[LeVeque, 2002]** R.J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge University Press, 2002. Chapitre 13 : problème de Riemann pour les équations d'Euler, structure des ondes.
