# Chapitre 8 : Cas test et post-traitement

## Introduction

La **verification** et la **validation** (V&V) sont deux etapes fondamentales en simulation numerique. La verification consiste a s'assurer que les equations discretisees sont resolues correctement (*"Do we solve the equations right?"*), tandis que la validation confronte les resultats a des donnees experimentales ou des solutions analytiques (*"Do we solve the right equations?"*).

Dans ce chapitre, nous nous concentrons sur la verification : chaque schema numerique doit etre teste sur un ensemble de cas test qui exercent differents aspects de la methode. Deux categories de cas test sont utilisees :

1. **Problemes de Riemann** (solutions discontinues) : ils testent la capacite du schema a capturer les chocs, les discontinuites de contact et les detentes. La solution exacte est obtenue par le solveur de Riemann exact (Ch. 2).
2. **Problemes lisses** (solutions regulieres) : ils permettent de mesurer l'ordre de convergence effectif du schema. La solution exacte est analytique et l'erreur doit decroitre comme $(\Delta x)^p$ ou $p$ est l'ordre theorique.

Chaque cas test est concu pour mettre en evidence une difficulte specifique : resolution de contact, robustesse en basse pression, dissipation des structures lisses, etc. Un schema qui fonctionne bien sur un cas peut echouer sur un autre, d'ou l'importance de tester systematiquement.

---

## 8.1 Tube de Sod (*Sod shock tube*)

Le tube de Sod est le cas test le plus classique de la dynamique des gaz compressibles. Il modelise l'eclatement d'une membrane separant deux chambres a des pressions differentes.

**Conditions initiales :**

$$
(\rho, u, p)_L = (1,\; 0,\; 1), \qquad (\rho, u, p)_R = (0.125,\; 0,\; 0.1)
$$

avec la discontinuite en $x_d = 0.5$, sur le domaine $[0, 1]$.

**Configuration :** $t_{\text{final}} = 0.2$, conditions aux limites transmissives, $\gamma = 1.4$.

**Scenario physique :** A $t = 0$, la membrane eclate. La difference de pression (rapport 10:1) genere trois ondes qui se propagent :

1. Une **detente** (eventail de rarefaction) vers la gauche, qui accelere le gaz et diminue progressivement sa pression.
2. Une **discontinuite de contact** qui separe deux regions a meme pression et meme vitesse mais a densites differentes.
3. Un **choc** vers la droite, qui comprime brutalement le gaz de droite.

**Ce que ce test verifie :**
- Capture correcte des trois types d'ondes du systeme d'Euler.
- Nettete du choc (nombre de mailles dans la zone de transition).
- Resolution de la discontinuite de contact (souvent plus diffusee que le choc par les schemas dissipatifs).
- Absence d'oscillations parasites pres des discontinuites.

**Reference :** [Sod, 1978].

---

## 8.2 Test de Lax

Le test de Lax est un probleme de Riemann plus severe que le tube de Sod, avec des etats initiaux asymetriques et une vitesse non nulle a gauche.

**Conditions initiales :**

$$
(\rho, u, p)_L = (0.445,\; 0.698,\; 3.528), \qquad (\rho, u, p)_R = (0.5,\; 0,\; 0.571)
$$

avec la discontinuite en $x_d = 0.5$, sur le domaine $[0, 1]$.

**Configuration :** $t_{\text{final}} = 0.14$, conditions aux limites transmissives, $\gamma = 1.4$.

**Scenario physique :** La vitesse non nulle a gauche ($u_L = 0.698$) cree des interactions plus fortes entre les ondes. Le choc resultant est plus intense que dans le cas de Sod, et la discontinuite de contact est plus marquee.

**Ce que ce test verifie :**
- Robustesse du schema face a des etats asymetriques avec vitesse non nulle.
- Capacite a traiter des chocs plus forts.
- Precision de la structure d'onde complete dans un regime plus exigeant.

---

## 8.3 Double detente (*123 problem*)

Ce cas test est un *stress test* qui pousse les schemas dans leurs retranchements en creant une zone de tres basse pression au centre du domaine.

**Conditions initiales :**

$$
(\rho, u, p)_L = (1,\; -2,\; 0.4), \qquad (\rho, u, p)_R = (1,\; 2,\; 0.4)
$$

avec la discontinuite en $x_d = 0.5$, sur le domaine $[0, 1]$.

**Configuration :** $t_{\text{final}} = 0.15$, conditions aux limites transmissives, $\gamma = 1.4$.

**Scenario physique :** Les deux etats ont des vitesses opposees ($u_L = -2$, $u_R = +2$) qui eloignent le gaz du centre. Cela genere deux detentes symetriques se propageant vers les bords, laissant au centre une zone de pression et de densite tres faibles.

**Ce que ce test verifie :**
- Robustesse en conditions de quasi-vide : la pression peut descendre tres bas au centre.
- Preservation de la positivite de la pression et de la densite.
- Certains schemas produisent des pressions negatives sur ce cas, ce qui provoque un crash du calcul.

**Danger :** Si la condition de formation du vide est satisfaite :

$$
u_R - u_L \geq \frac{2}{\gamma - 1}(a_L + a_R)
$$

alors un vide se forme et le solveur de Riemann exact leve une erreur (voir Ch. 2, SS 2.6). Dans le cas present, cette condition n'est pas atteinte mais on s'en approche.

---

## 8.4 Contact stationnaire (*stationary contact*)

Ce cas test isole la **diffusion numerique** d'un schema en eliminant toute autre phenomenologie (pas de choc, pas de detente).

**Conditions initiales :**

$$
(\rho, u, p)_L = (1,\; 0,\; 1), \qquad (\rho, u, p)_R = (0.125,\; 0,\; 1)
$$

avec la discontinuite en $x_0 = 0.5$.

**Solution exacte :** La solution est **stationnaire pour tout temps**. La vitesse est nulle, la pression est uniforme, et la discontinuite de densite reste immobile a $x = 0.5$ avec un saut de $\rho = 1$ a $\rho = 0.125$. Tout etalement du profil de densite dans la solution numerique est exclusivement du a la diffusion numerique du schema.

**Configuration :** Domaine $[0, 1]$, conditions aux limites transmissives, $t_{\text{final}} = 0.5$ (temps long pour amplifier la diffusion), $\gamma = 1.4$.

**Pourquoi ce cas est revelateur :** La dissipation numerique d'un flux sur l'onde de contact est proportionnelle a $|\tilde{\lambda}_2|$ pour Roe, ou au traitement de l'onde intermediaire pour HLL/HLLC. Ici $\lambda_2 = u = 0$, ce qui signifie :

- **Godunov, HLLC, Roe** : la dissipation sur l'onde de contact est proportionnelle a $|u| = 0$. Le contact reste **parfaitement raide** (zero diffusion).
- **HLL** : n'a pas d'onde de contact dans son modele. La dissipation est controlee par $S_L$ et $S_R$ qui sont non nuls ($S_L = -a$, $S_R = +a$). Le contact est **fortement etale**.
- **Rusanov** : la dissipation est $S_{\max} = |u| + a = a > 0$. Meme diffusion que HLL.

**Ce que ce test verifie :**
- Capacite du flux a resoudre l'onde de contact sans diffusion parasite.
- Difference fondamentale entre les flux a 2 ondes (HLL) et a 3 ondes (HLLC, Roe, Godunov).

---

## 8.5 Quasi-vide (*near vacuum*)

Ce cas test est une version extreme de la double detente (SS 8.3), avec des vitesses proches de la limite de formation du vide.

**Conditions initiales :**

$$
(\rho, u, p)_L = (1,\; -3.5,\; 0.4), \qquad (\rho, u, p)_R = (1,\; 3.5,\; 0.4)
$$

avec la discontinuite en $x_0 = 0.5$.

**Proximite du vide :** La vitesse du son dans les deux etats est $a = \sqrt{\gamma p / \rho} = \sqrt{0.56} \approx 0.748$. Le seuil de formation du vide est :

$$
\Delta u_{\text{vide}} = \frac{2}{\gamma - 1}(a_L + a_R) = 5 \times 1.497 = 7.483
$$

Ici $\Delta u = u_R - u_L = 7.0$, soit **93%** du seuil. La pression de la zone etoile est extremement faible : $p^* \sim 10^{-13}$.

**Comparaison avec la double detente (SS 8.3) :** La double detente utilise $u = \pm 2$ ($\Delta u = 4$, soit 53% du seuil). Le quasi-vide est un cas bien plus severe.

**Configuration :** Domaine $[0, 1]$, conditions aux limites transmissives, $t_{\text{final}} = 0.1$, $\gamma = 1.4$.

**Ce que ce test verifie :**
- **Preservation de la positivite** de la pression et de la densite. Le plancher de positivite du solveur est mis a rude epreuve.
- Robustesse des flux numeriques avec des densites/pressions quasi nulles.
- Correctement du calcul des eventails de detente pres de la limite du vide.

**Comportement observe :**
- Certains schemas (Roe sans correction) peuvent generer des pressions negatives et diverger sans plancher de positivite.
- Les schemas les plus dissipatifs (Rusanov) sont paradoxalement les plus robustes car ils maintiennent une pression plancher plus elevee.

---

## 8.6 Collision de deux chocs (*two shocks*, Toro test 4)

Ce cas test met en jeu deux chocs forts qui se propagent l'un vers l'autre et entrent en collision pres du centre du domaine.

**Conditions initiales :**

$$
(\rho, u, p)_L = (5.99924,\; 19.5975,\; 460.894), \qquad (\rho, u, p)_R = (5.99242,\; -6.19633,\; 46.0950)
$$

avec la discontinuite en $x_0 = 0.4$.

**Structure des ondes :** Les deux etats ont des vitesses dirigees vers l'interieur du domaine ($u_L > 0$, $u_R < 0$), ce qui genere une compression violente :

1. **Choc gauche** se propageant vers la gauche.
2. **Discontinuite de contact** au centre.
3. **Choc droit** se propageant vers la droite.

L'etat intermediaire atteint une pression tres elevee ($p^* \approx 1700$) et une densite elevee, creant des conditions post-choc extremes.

**Configuration :** Domaine $[0, 1]$, conditions aux limites transmissives, $t_{\text{final}} = 0.035$, $\gamma = 1.4$.

**Ce cas test est le symetrique de la double detente (SS 8.3)** : au lieu d'etirer le gaz (detentes), on le comprime (chocs). La double detente teste la robustesse en basse pression ; Toro test 4 teste la robustesse en haute pression.

**Ce que ce test verifie :**
- Capture correcte de l'interaction choc-choc.
- Resolution de la discontinuite de contact entre les deux chocs (densite et vitesse y varient fortement).
- Robustesse sous conditions post-choc extremes (forte pression et densite).

**Differences observees entre schemas :**
- **Roe** et **Roe-NC** donnent des resultats identiques (pas de detente transsonique dans ce cas).
- **Rusanov** est nettement plus dissipatif, surtout sur le contact.
- **HLLC** offre un bon compromis, legerement inferieur a Roe sur ce cas particulier.

**Reference :** [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e ed., Springer, Test 4, p. 334.

---

## 8.7 Interaction choc/onde entropique (*Shu-Osher*)

Ce cas test est le seul qui n'est pas un probleme de Riemann pur. Il met en evidence l'apport des schemas d'ordre eleve de maniere spectaculaire.

**Conditions initiales** (domaine $[-5, 5]$) :

- A gauche de $x = -4$ (etat post-choc) : $(\rho, u, p) = (3.857143,\; 2.629369,\; 10.33333)$
- A droite de $x = -4$ : $\rho = 1 + 0.2\sin(5x)$, $u = 0$, $p = 1$

Un choc de Mach 3 se propage vers la droite dans un milieu dont la densite presente des perturbations sinusoidales.

**Phenomenologie :** L'interaction entre le choc et les perturbations de densite genere une structure complexe derriere le choc :
- Des oscillations haute frequence de densite qui sont **physiques** (pas des artefacts numeriques).
- Ces oscillations sont amplifiees par le passage du choc (mecanisme de Richtmyer-Meshkov).

**Solution de reference :** Il n'existe pas de solution analytique. La reference est une simulation a haute resolution (WENO5-Z + HLLC, 2000 cellules) interpolee sur le maillage demande.

**Configuration :** Domaine $[-5, 5]$, conditions aux limites transmissives, $t_{\text{final}} = 1.8$, $\gamma = 1.4$.

**Ce que ce test verifie :**
- **Ordre 1 (HLLC, Rusanov)** : le choc est capture mais toutes les oscillations post-choc sont lissees.
- **Ordre 2 (MUSCL-HLLC)** : les oscillations de grande echelle apparaissent mais les details fins sont amortis.
- **Ordre 5 (WENO5-HLLC)** : les oscillations sont capturees fidelement, tres proches de la reference.

C'est le meilleur cas test pour justifier le surcout des schemas d'ordre eleve.

**Reference :** [Shu, Osher, 1989] C.-W. Shu, S. Osher, *Efficient implementation of essentially non-oscillatory shock-capturing schemes, II*, J. Comput. Phys., 83(1), pp. 32-78.

---

## 8.8 Onde entropique (*entropy wave*)

Ce cas test lisse permet de verifier l'ordre de convergence d'un schema sur un mode purement entropique.

**Conditions initiales :**

$$
\rho(x, 0) = 1 + \varepsilon \sin(2\pi x), \qquad u(x, 0) = u_0 = 1, \qquad p(x, 0) = p_0 = 1
$$

avec $\varepsilon = 0.2$ par defaut.

**Solution exacte :** La perturbation de densite est simplement advectee a la vitesse $u_0$ :

$$
\rho(x, t) = 1 + \varepsilon \sin\bigl(2\pi(x - u_0 t)\bigr), \qquad u(x, t) = u_0, \qquad p(x, t) = p_0
$$

**Configuration :** Domaine $[0, 1]$, conditions aux limites periodiques, $t_{\text{final}} = 1.0$ (un aller-retour complet du domaine), $\gamma = 1.4$.

**Lien avec l'analyse des ondes :** Ce cas test excite uniquement le mode entropique du systeme d'Euler (Ch. 7, SS 7.2) : seule la densite varie, tandis que la vitesse et la pression restent constantes. L'onde se propage a la vitesse $u_0$ (valeur propre $\lambda_2 = u$).

**Ce que ce test verifie :**
- Dissipation numerique des structures lisses : apres un temps final $t = 1$, la sinusoide doit etre restituee fidalement. Un schema dissipatif amortit l'amplitude.
- **Verification de l'ordre de convergence** : sur ce cas lisse, un schema d'ordre $N$ doit produire une erreur proportionnelle a $(\Delta x)^N$. C'est le cas test principal pour confirmer l'ordre theorique.

---

## 8.9 Onde acoustique (*acoustic wave*)

Ce cas test lisse permet de verifier la convergence sur tous les champs simultanement, en excitant un mode acoustique couple.

**Conditions initiales :** Petite perturbation isentropique alignee sur le vecteur propre acoustique droit (Ch. 1, SS 1.4) :

$$
\delta\rho = \varepsilon \sin(2\pi x), \qquad \delta u = \frac{a_0}{\rho_0}\,\varepsilon \sin(2\pi x), \qquad \delta p = a_0^2\,\varepsilon \sin(2\pi x)
$$

autour de l'etat de base $(\rho_0, u_0, p_0) = (1, 0, 1)$, avec $\varepsilon = 10^{-4}$ (suffisamment petit pour que la solution linearisee soit valide) et $a_0 = \sqrt{\gamma p_0 / \rho_0}$.

**Solution exacte :** La perturbation se propage a la vitesse du son $a_0$ sans deformation :

$$
\rho(x, t) = \rho_0 + \varepsilon \sin\bigl(2\pi(x - a_0 t)\bigr)
$$

et de meme pour $u$ et $p$ avec les coefficients du vecteur propre.

**Configuration :** Domaine $[0, 1]$, conditions aux limites periodiques, $t_{\text{final}} = 1/a_0$ (une traversee du domaine), $\gamma = 1.4$.

**Difference avec l'onde entropique :** Ici, les trois champs ($\rho$, $u$, $p$) varient simultanement de maniere couplee. Un schema qui converge bien sur la densite seule (onde entropique) pourrait avoir un comportement different sur les ondes acoustiques, ou les erreurs de dispersion affectent la phase de propagation.

**Ce que ce test verifie :**
- Propagation correcte des ondes veritablement non lineaires.
- Convergence sur tous les champs simultanement (pas seulement la densite).
- Erreurs de dispersion numerique (vitesse de phase).

---

## 8.10 Analyse de convergence

L'etude de convergence en maillage est l'outil principal pour verifier qu'un schema atteint bien son ordre de precision theorique.

### Normes d'erreur

Pour une variable $f$ (par exemple $\rho$), on mesure l'ecart entre la solution numerique $f_i$ et la solution exacte $f_i^{\text{exact}}$ a l'aide des normes suivantes :

$$
\|e\|_1 = \sum_{i=1}^{N} |f_i - f_i^{\text{exact}}| \, \Delta x
$$

$$
\|e\|_2 = \sqrt{\sum_{i=1}^{N} (f_i - f_i^{\text{exact}})^2 \, \Delta x}
$$

$$
\|e\|_\infty = \max_{1 \leq i \leq N} |f_i - f_i^{\text{exact}}|
$$

La norme $L_1$ mesure l'erreur globale integree, la norme $L_2$ penalise davantage les grandes erreurs locales, et la norme $L_\infty$ capture l'erreur maximale ponctuelle.

### Protocole de convergence

1. Choisir un cas test lisse (onde entropique ou acoustique) avec solution exacte analytique.
2. Fixer le nombre de CFL a une valeur constante (typiquement $\text{CFL} = 0.9$). Cela garantit que $\Delta t \propto \Delta x$, et que l'erreur temporelle reste du meme ordre que l'erreur spatiale.
3. Lancer la simulation pour plusieurs resolutions croissantes, par exemple $N = 50, 100, 200, 400, 800$.
4. Pour chaque resolution, calculer l'erreur par rapport a la solution exacte.
5. Estimer l'ordre $p$ par regression lineaire en echelle log-log :

$$
\log(\|e\|) = p \, \log(\Delta x) + c
$$

### Ordres attendus

Sur les cas test **lisses**, les ordres de convergence attendus sont :

| Schema | Ordre theorique |
|--------|----------------|
| Godunov, Rusanov, HLL, HLLC, Roe | 1 |
| MUSCL, ENO2, Lax-Wendroff | 2 |
| WENO3 | 3 |
| WENO5 | 5 |

Sur les problemes de **Riemann** (solutions discontinues), l'ordre de convergence global chute a environ 1, quel que soit l'ordre du schema. Cela est du a la presence de discontinuites qui limitent la regularite de la solution. L'interet des schemas d'ordre eleve sur les problemes discontinus reside dans la resolution plus nette des structures (moins de dissipation numerique) plutot que dans un taux de convergence plus eleve.

---

## 8.11 Implementation

### `euler1d/test_cases.py`

Ce module contient des **fonctions factory** qui retournent chacune un objet `SimulationConfig` completement defini (gaz, maillage, temps, probleme, conditions aux limites). Chaque fonction correspond a un des cas test decrits ci-dessus :

- `sod_shock_tube(n_cells)` -- SS 8.1
- `lax_test(n_cells)` -- SS 8.2
- `double_rarefaction(n_cells)` -- SS 8.3
- `entropy_wave(n_cells, epsilon)` -- SS 8.4
- `acoustic_wave(n_cells, epsilon)` -- SS 8.5

Les cas lisses (`SmoothProblem`) fournissent une fonction `exact_fn(x, t, gamma)` qui retourne la solution analytique, utilisee pour le calcul des erreurs et des ordres de convergence.

### `euler1d/results.py`

Ce module fournit les outils de post-traitement :

- `compute_exact_solution(config)` : calcule la solution exacte sur le maillage, soit par le solveur de Riemann exact (problemes discontinus), soit par la solution analytique (problemes lisses).
- `compute_errors(result, exact)` : calcule les normes $L_1$, $L_2$, $L_\infty$ pour $\rho$, $u$ et $p$ (SS 8.6).
- `convergence_study(make_config, scheme, n_cells_list)` : lance une etude de convergence complete a CFL constant et retourne un `DataFrame` avec les erreurs pour chaque resolution.
- `estimate_order(dx_values, error_values)` : estime l'ordre de convergence par regression log-log.

---

## References

- **[Sod, 1978]** G.A. Sod, *A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws*, Journal of Computational Physics, 27(1), pp. 1-31, 1978.
- **[Toro, 2009]** E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd edition, Springer, 2009. Ch. 6 (cas test numeriques), Ch. 17 (methodes d'ordre eleve).
