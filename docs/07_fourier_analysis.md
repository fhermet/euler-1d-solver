# Chapitre 7 : Analyse de Fourier

## Introduction

Dans les regions lisses de l'ecoulement, les deux principales sources d'erreur d'un schema numerique sont la **dissipation numerique** (amortissement de l'amplitude des ondes) et la **dispersion numerique** (erreur sur la vitesse de phase). L'analyse de Fourier permet de quantifier ces deux phenomenes en etudiant comment le schema transforme chaque mode de Fourier individuellement.

Cette analyse est **independante du cas test** : on linearise les equations autour d'un etat uniforme et on etudie la reponse du schema complet (reconstruction spatiale + flux numerique + integration temporelle) a une perturbation sinusoidale de nombre d'onde $k$ donne. Le resultat est le **facteur d'amplification** $G(\theta)$, nombre complexe qui encode a la fois l'amortissement et le dephasage introduits par le schema.

L'implementation se trouve dans `euler1d/fourier_analysis.py`.

---

## 7.1 Linearisation

On gele les coefficients autour d'un etat uniforme $(\rho_0, u_0, p_0) = (1, 1, 1)$. Les equations d'Euler 1D en variables primitives $W = (\rho, u, p)^T$ s'ecrivent sous forme linearisee :

$$\frac{\partial W}{\partial t} + A_0 \frac{\partial W}{\partial x} = 0$$

ou $A_0$ est la matrice jacobienne du systeme primitif evaluee a l'etat de base :

$$A_0 = \begin{pmatrix} u_0 & \rho_0 & 0 \\ 0 & u_0 & 1/\rho_0 \\ 0 & \rho_0 c_0^2 & u_0 \end{pmatrix}$$

avec $c_0 = \sqrt{\gamma p_0 / \rho_0}$ la vitesse du son. Cette matrice admet trois valeurs propres reelles :

- $\lambda_1 = u_0 - c_0$ (onde acoustique gauche),
- $\lambda_2 = u_0$ (onde entropique),
- $\lambda_3 = u_0 + c_0$ (onde acoustique droite).

Chaque mode propre se propage independamment a sa vitesse propre. C'est cette propriete qui permet d'analyser la reponse du schema mode par mode.

---

## 7.2 Modes propres

Le systeme linearise admet trois modes independants, chacun associe a un vecteur propre droit (direction de perturbation) et un vecteur propre gauche (direction de projection).

### Onde entropique

Perturbation alignee sur le deuxieme vecteur propre :

$$\delta W = \varepsilon \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$$

Seule la densite varie. La perturbation est advectee a la vitesse $u_0$. La pression et la vitesse restent constantes.

**Vecteur propre gauche** (pour projection) :

$$\ell_{\text{entropie}} = \left(1,\; 0,\; -\frac{1}{c_0^2}\right)$$

On verifie : $\ell_{\text{entropie}} \cdot \delta W = \varepsilon \times 1 = \varepsilon$.

### Onde acoustique droite

Perturbation alignee sur le troisieme vecteur propre :

$$\delta W = \varepsilon \begin{pmatrix} 1 \\ c_0 / \rho_0 \\ c_0^2 \end{pmatrix}$$

Tous les champs varient. La perturbation se propage a la vitesse $u_0 + c_0$.

**Vecteur propre gauche** :

$$\ell_{\text{acoustique}+} = \left(0,\; \frac{\rho_0}{2 c_0},\; \frac{1}{2 c_0^2}\right)$$

On verifie : $\ell_{\text{acoustique}+} \cdot \delta W = \varepsilon \left(\frac{\rho_0}{2c_0} \cdot \frac{c_0}{\rho_0} + \frac{c_0^2}{2c_0^2}\right) = \varepsilon \left(\frac{1}{2} + \frac{1}{2}\right) = \varepsilon$.

### Onde acoustique gauche

Perturbation alignee sur le premier vecteur propre :

$$\delta W = \varepsilon \begin{pmatrix} 1 \\ -c_0 / \rho_0 \\ c_0^2 \end{pmatrix}$$

Se propage a la vitesse $u_0 - c_0$.

**Vecteur propre gauche** :

$$\ell_{\text{acoustique}-} = \left(0,\; -\frac{\rho_0}{2 c_0},\; \frac{1}{2 c_0^2}\right)$$

Les vecteurs propres gauches forment les lignes de $R^{-1}$ (inverse de la matrice des vecteurs propres droits). Ils sont orthogonaux aux autres modes : projeter la sortie sur $\ell$ isole la composante du mode excite et filtre les contributions parasites des deux autres modes.

---

## 7.3 Facteur d'amplification $G(\theta)$

Pour un nombre d'onde $k$, on definit le **nombre d'onde reduit** :

$$\theta = k \Delta x$$

qui varie de $0$ (mode constant) a $\pi$ (mode de Nyquist, 2 points par longueur d'onde).

On injecte une perturbation sinusoidale alignee sur un mode propre :

$$W(x, 0) = W_0 + A_0 \, e^{ikx} \, \mathbf{r}$$

ou $\mathbf{r}$ est le vecteur propre du mode choisi et $A_0$ l'amplitude initiale. Apres **un pas de temps complet** (incluant tous les etages Runge-Kutta), la sortie projetee sur le vecteur propre gauche donne :

$$\text{sortie} = G(\theta) \, A_0 \, e^{ikx}$$

Le facteur d'amplification $G(\theta)$ est un nombre complexe qui encode deux informations :

### Module : dissipation

$$\lvert G(\theta) \rvert = \text{facteur d'amortissement}$$

- $\lvert G \rvert = 1$ : pas d'amortissement (schema ideal).
- $\lvert G \rvert < 1$ : le mode est amorti. Le schema est **dissipatif**.
- $\lvert G \rvert > 1$ : le mode est amplifie. Le schema est **instable** pour ce mode.

### Phase : dispersion

$$\arg(G) = \varphi_{\text{num}} = \text{dephasage numerique}$$

La phase exacte apres un pas de temps est :

$$\varphi_{\text{exact}} = -\sigma \theta$$

ou $\sigma = \lambda \Delta t / \Delta x$ est le nombre de Courant du mode ($\lambda$ etant la vitesse propre).

Le **rapport de phase** mesure l'erreur de dispersion :

$$\frac{\varphi_{\text{num}}}{\varphi_{\text{exact}}}$$

- Rapport $= 1$ : vitesse de phase correcte (schema ideal).
- Rapport $> 1$ : l'onde numerique va **trop vite** (erreur de phase avancee).
- Rapport $< 1$ : l'onde numerique va **trop lentement** (erreur de phase retardee).

---

## 7.4 Methodologie d'extraction

Le facteur $G(\theta)$ est un nombre complexe : il faut determiner a la fois sa partie reelle et sa partie imaginaire. Pour cela, le code utilise une **technique a double perturbation** (cosinus et sinus).

### Pourquoi deux simulations ?

Une seule simulation avec une perturbation $\cos(kx)$ donne en sortie un signal contenant a la fois $\cos(kx)$ et $\sin(kx)$ (car $G$ est complexe). On peut extraire les coefficients $a_{cc}$ et $a_{cs}$ par projection, mais cela ne suffit pas a separer $\text{Re}(G)$ et $\text{Im}(G)$ : on a deux inconnues et seulement deux equations reliees entre elles. La deuxieme simulation avec $\sin(kx)$ fournit les equations supplementaires necessaires.

### Procedure detaillee

Pour chaque mode $m$ (nombre d'onde $k = 2\pi m / L$, nombre d'onde reduit $\theta = k \Delta x$) :

**Etape 1 -- Perturbation cosinus.** On initialise :

$$W_c(x) = W_0 + \varepsilon \, \mathbf{r} \, \cos(kx)$$

On effectue un pas de temps complet avec le schema. On projette la sortie sur le vecteur propre gauche $\ell$ pour obtenir le signal scalaire $\delta_c(x)$.

**Etape 2 -- Perturbation sinus.** Meme chose avec :

$$W_s(x) = W_0 + \varepsilon \, \mathbf{r} \, \sin(kx)$$

On obtient le signal scalaire $\delta_s(x)$.

**Etape 3 -- Projection modale.** On extrait les coefficients de Fourier par produit scalaire discret :

$$a_{cc} = \frac{2}{N} \sum_{j=1}^{N} \delta_c(x_j) \cos(kx_j), \quad a_{cs} = \frac{2}{N} \sum_{j=1}^{N} \delta_c(x_j) \sin(kx_j)$$

$$a_{sc} = \frac{2}{N} \sum_{j=1}^{N} \delta_s(x_j) \cos(kx_j), \quad a_{ss} = \frac{2}{N} \sum_{j=1}^{N} \delta_s(x_j) \sin(kx_j)$$

**Etape 4 -- Reconstruction de $G$.** Les parties reelle et imaginaire du facteur d'amplification sont :

$$\text{Re}(G) = \frac{a_{cc} + a_{ss}}{2 A_0}, \quad \text{Im}(G) = \frac{a_{sc} - a_{cs}}{2 A_0}$$

ou $A_0 = \varepsilon$ est l'amplitude d'entree (verifiee par $\ell \cdot \varepsilon \mathbf{r} = \varepsilon$).

**Etape 5 -- Module et phase.**

$$\lvert G \rvert = \sqrt{\text{Re}(G)^2 + \text{Im}(G)^2}, \quad \varphi = \text{atan2}(\text{Im}(G),\, \text{Re}(G))$$

**Etape 6 -- Post-traitement de la phase.** La phase est depliee avec `np.unwrap` pour eviter les sauts de $2\pi$. Le rapport de phase $\varphi_{\text{num}} / \varphi_{\text{exact}}$ est mis a `NaN` lorsque $\lvert G \rvert < 10^{-10}$ (mode trop amorti pour que la phase ait un sens physique) ou lorsque $\lvert \varphi_{\text{exact}} \rvert < 10^{-12}$ (mode quasi-stationnaire).

---

## 7.5 Interpretation des courbes

### Courbe de dissipation $\lvert G(\theta) \rvert$

- **Schema ideal** : $\lvert G \rvert = 1$ pour tout $\theta$ (courbe horizontale).
- **Schemas d'ordre 1** (Rusanov, HLL, HLLC, Godunov) : $\lvert G \rvert$ chute fortement des les moyennes frequences. Ces schemas sont tres dissipatifs : ils amortissent significativement les ondes courtes. Cela lisse les discontinuites mais diffuse aussi les structures fines.
- **Lax-Wendroff** : $\lvert G \rvert \approx 1$ sur une large plage de $\theta$. Le schema est peu dissipatif, ce qui preserv les ondes mais peut generer des **oscillations** pres des discontinuites (pas d'amortissement des hautes frequences parasites).
- **Schemas d'ordre eleve** (MUSCL, WENO) : $\lvert G \rvert$ reste proche de 1 jusqu'a des $\theta$ moderes, puis chute pour les hautes frequences. Le compromis est meilleur : les ondes physiques sont preservees, et seules les frequences mal resolues sont amorties.
- **Instabilite** : si $\lvert G \rvert > 1$ pour un $\theta$ donne, le mode est amplifie a chaque pas de temps. Cela indique un schema instable pour ce CFL.
- Une chute brutale a $\theta = \pi$ signifie que le schema ne resout que les basses frequences.

### Courbe de dispersion $\varphi_{\text{num}} / \varphi_{\text{exact}}$

- **Schema ideal** : rapport $= 1$ pour tout $\theta$.
- **Rapport $> 1$** a un $\theta$ donne : l'onde numerique se propage plus vite que la vitesse exacte. Cela produit une **erreur de phase avancee** (l'onde arrive trop tot).
- **Rapport $< 1$** : l'onde se propage trop lentement (**erreur de phase retardee**).
- **Lax-Wendroff** : $\lvert G \rvert$ est proche de 1, mais le rapport de phase s'ecarte significativement de 1 aux hautes frequences. C'est la signature d'un schema **dispersif** : il ne dissipe pas les ondes mais les fait voyager a la mauvaise vitesse, ce qui genere des oscillations en amont et en aval des discontinuites.
- **Schemas d'ordre eleve** : le rapport de phase reste proche de 1 sur une plage de $\theta$ d'autant plus large que l'ordre est eleve.

### Signatures typiques

| Schema | Dissipation | Dispersion |
|--------|-------------|------------|
| Ordre 1 (Rusanov, Godunov...) | $\lvert G \rvert$ chute fortement | Phase peu pertinente (mode trop amorti) |
| Lax-Wendroff | $\lvert G \rvert \approx 1$ | Fort ecart de phase aux hautes frequences |
| MUSCL + limiteur | $\lvert G \rvert$ proche de 1 en basses freq. | Bon rapport de phase en basses freq. |
| WENO3/WENO5 | $\lvert G \rvert \approx 1$ jusqu'a $\theta$ modere | Rapport de phase $\approx 1$ sur une large plage |

---

## 7.6 Implementation

L'implementation se trouve dans le module `euler1d/fourier_analysis.py`. Deux fonctions principales :

### `_single_step(U, scheme, gas, dx, dt)`

Avance l'etat $U$ d'un pas de temps en utilisant l'integrateur temporel par defaut du schema. Cette fonction reproduit la boucle Runge-Kutta complete (et non seulement l'operateur spatial) car l'analyse de Fourier doit mesurer le comportement du schema complet, integration temporelle incluse. Les cinq methodes RK sont supportees :

| Methode | Etages | Schemas concernes |
|---------|--------|-------------------|
| RK1 | 1 | Euler explicite, Lax-Wendroff |
| RK2 | 2 | MUSCL, ENO2 |
| RK3 (SSP) | 3 | WENO3 |
| RK4 | 4 | JST |
| RK5 (Dormand-Prince) | 6 | WENO5 |

### `compute_amplification(scheme, cfl, n_cells, gamma, n_modes, wave_type)`

Calcule le facteur d'amplification $G(\theta)$ pour un schema donne en utilisant la technique a double perturbation decrite en [section 7.4](#74-methodologie-dextraction). Parametres principaux :

- `wave_type` : `"entropie"` (onde de densite a vitesse $u_0$) ou `"acoustique"` (onde acoustique droite a vitesse $u_0 + c_0$).
- `n_cells` : nombre de cellules du maillage periodique (256 par defaut).
- `n_modes` : nombre de modes de Fourier analyses (de $m = 1$ a $m = n_{\text{modes}}$).

Le dictionnaire retourne contient :

- `theta` : tableau des nombres d'onde reduits $\theta = 2\pi m / N$.
- `abs_G` : tableau des $\lvert G(\theta) \rvert$ (dissipation).
- `phase_ratio` : tableau des $\varphi_{\text{num}} / \varphi_{\text{exact}}$ (dispersion, `NaN` si le mode est trop amorti).

---

## References

- [Toro, 2009] E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3e edition, Springer. Chapitre 13 : analyse de Fourier des schemas de volumes finis.
- [Hirsch, 2007] C. Hirsch, *Numerical Computation of Internal and External Flows*, 2e edition, Elsevier. Chapitre 8 : analyse de stabilite et proprietes dissipatives/dispersives.
