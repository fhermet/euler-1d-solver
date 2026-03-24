# Bibliographie

Les references ci-dessous sont citees dans les chapitres 1 a 8 de la documentation ainsi que dans les docstrings du code source. Elles couvrent les fondements theoriques des equations d'Euler, les solveurs de Riemann exacts et approches, les methodes de reconstruction d'ordre eleve, les schemas d'integration temporelle et l'analyse de Fourier des schemas de volumes finis.

---

**[Batten et al., 1997]** P. Batten, N. Clarke, C. Lambert, D.M. Causon — *"On the Choice of Wavespeeds for the HLLC Riemann Solver"*, SIAM Journal on Scientific Computing, 18(6), pp. 1553-1570, 1997. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.4 — estimation des vitesses d'onde pour le schema HLLC).

**[Borges et al., 2008]** R. Borges, M. Carmona, B. Costa, W.S. Don — *"An Improved Weighted Essentially Non-Oscillatory Scheme for Hyperbolic Conservation Laws"*, Journal of Computational Physics, 227, pp. 3191-3211, 2008. Chapitres pertinents : [Chapitre 5](05_reconstruction.md) (§5.3 — variante WENO-Z pour WENO3, §5.4 — variante WENO-Z pour WENO5).

**[Davis, 1988]** S.F. Davis — *"Simplified Second-Order Godunov-Type Methods"*, SIAM Journal on Scientific and Statistical Computing, 9(3), pp. 445-473, 1988. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.3 — estimation des vitesses d'onde pour le schema HLL).

**[Dormand, Prince, 1980]** J.R. Dormand, P.J. Prince — *"A Family of Embedded Runge-Kutta Formulae"*, Journal of Computational and Applied Mathematics, 6(1), pp. 19-26, 1980. Chapitres pertinents : [Chapitre 6](06_time_integration.md) (§6.6 — methode Dormand-Prince RK5 a 6 etages pour WENO5).

**[Godunov, 1959]** S.K. Godunov — *"A Difference Method for the Numerical Calculation of Discontinuous Solutions of the Equations of Hydrodynamics"*, Matematicheskii Sbornik, 47(3), pp. 271-306, 1959. Chapitres pertinents : [Chapitre 2](02_riemann_problem.md) (utilisation du probleme de Riemann comme brique de base), [Chapitre 4](04_flux_schemes.md) (§4.1 — schema de Godunov), [Chapitre 5](05_reconstruction.md) (theoreme de Godunov sur les schemas lineaires monotones).

**[Gottlieb, Shu, Tadmor, 2001]** S. Gottlieb, C.-W. Shu, E. Tadmor — *"Strong Stability-Preserving High-Order Time Discretization Methods"*, SIAM Review, 43(1), pp. 89-112, 2001. Chapitres pertinents : [Chapitre 6](06_time_integration.md) (§6.5 — absence de methode SSP d'ordre 4 a 4 etages, proprietes SSP generales).

**[Harten, Hyman, 1983]** A. Harten, J.M. Hyman — *"Self Adjusting Grid Methods for One-Dimensional Hyperbolic Conservation Laws"*, Journal of Computational Physics, 50, pp. 235-269, 1983. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.5 — correction entropique de Harten pour le schema de Roe).

**[Harten, Lax, van Leer, 1983]** A. Harten, P.D. Lax, B. van Leer — *"On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws"*, SIAM Review, 25(1), pp. 35-61, 1983. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.3 — schema HLL a deux ondes).

**[Harten et al., 1987]** A. Harten, B. Engquist, S. Osher, S.R. Chakravarthy — *"Uniformly High Order Accurate Essentially Non-Oscillatory Schemes, III"*, Journal of Computational Physics, 71, pp. 231-303, 1987. Chapitres pertinents : [Chapitre 5](05_reconstruction.md) (§5.2 — reconstruction ENO2, selection de stencil).

**[Hirsch, 2007]** C. Hirsch — *"Numerical Computation of Internal and External Flows"*, 2e edition, Elsevier, 2007. Chapitres pertinents : [Chapitre 7](07_fourier_analysis.md) (analyse de stabilite, proprietes dissipatives et dispersives des schemas).

**[Jameson, Schmidt, Turkel, 1981]** A. Jameson, W. Schmidt, E. Turkel — *"Numerical Solutions of the Euler Equations by Finite Volume Methods Using Runge-Kutta Time Stepping Schemes"*, AIAA Paper 81-1259, 1981. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.8 — schema JST avec dissipation artificielle adaptative).

**[Jiang, Shu, 1996]** G.-S. Jiang, C.-W. Shu — *"Efficient Implementation of Weighted ENO Schemes"*, Journal of Computational Physics, 126, pp. 202-228, 1996. Chapitres pertinents : [Chapitre 5](05_reconstruction.md) (§5.3 — WENO3 variante JS, §5.4 — WENO5 variante JS, poids non lineaires et indicateurs de regularite).

**[Laney, 1998]** C.B. Laney — *"Computational Gasdynamics"*, Cambridge University Press, 1998. Chapitres pertinents : [Chapitre 1](01_euler_equations.md) (derivation physique des equations d'Euler, thermodynamique du gaz parfait).

**[Liou, 1996]** M.-S. Liou — *"A Sequel to AUSM: AUSM+"*, Journal of Computational Physics, 129, pp. 364-382, 1996. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.9 — schema AUSM+, splitting convection/pression en nombre de Mach).

**[Liou, 2006]** M.-S. Liou — *"A Sequel to AUSM, Part II: AUSM+-up for All Speeds"*, Journal of Computational Physics, 214, pp. 137-170, 2006. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.9 — schema AUSM+-up, corrections de dissipation en Mach et pression).

**[Lax, 1954]** P.D. Lax — *"Weak Solutions of Nonlinear Hyperbolic Equations and Their Numerical Computation"*, Communications on Pure and Applied Mathematics, 7, pp. 159-193, 1954. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.6 — schema de Lax-Friedrichs global).

**[Lax, Wendroff, 1960]** P.D. Lax, B. Wendroff — *"Systems of Conservation Laws"*, Communications on Pure and Applied Mathematics, 13, pp. 217-237, 1960. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.7 — schema de Lax-Wendroff, theoreme de Lax-Wendroff sur la convergence des schemas conservatifs).

**[LeVeque, 2002]** R.J. LeVeque — *"Finite Volume Methods for Hyperbolic Problems"*, Cambridge University Press, 2002. Chapitres pertinents : [Chapitre 1](01_euler_equations.md) (lois de conservation, hyperbolicite), [Chapitre 2](02_riemann_problem.md) (probleme de Riemann pour les equations d'Euler), [Chapitre 3](03_finite_volume.md) (methode des volumes finis).

**[Richtmyer, 1963]** R.D. Richtmyer — *"A Survey of Difference Methods for Non-Steady Fluid Dynamics"*, NCAR Technical Note 63-2, 1963. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.7 — formulation predicteur-correcteur de Richtmyer en deux etapes pour le schema de Lax-Wendroff).

**[Roe, 1981]** P.L. Roe — *"Approximate Riemann Solvers, Parameter Vectors, and Difference Schemes"*, Journal of Computational Physics, 43(2), pp. 357-372, 1981. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.5 — schema de Roe, linearisation exacte, moyennes de Roe).

**[Rusanov, 1961]** V.V. Rusanov — *"The Calculation of the Interaction of Non-Stationary Shock Waves with Barriers"*, Zhurnal Vychislitel'noi Matematiki i Matematicheskoi Fiziki, 1(2), pp. 267-279, 1961. Chapitres pertinents : [Chapitre 4](04_flux_schemes.md) (§4.2 — schema de Rusanov, Lax-Friedrichs local).

**[Shu, Osher, 1988]** C.-W. Shu, S. Osher — *"Efficient Implementation of Essentially Non-Oscillatory Shock-Capturing Schemes"*, Journal of Computational Physics, 77(2), pp. 439-471, 1988. Chapitres pertinents : [Chapitre 6](06_time_integration.md) (§6.4 — methode SSP-RK3 a 3 etages, ecriture de Shu-Osher en combinaisons convexes).

**[Shu, Osher, 1989]** C.-W. Shu, S. Osher — *"Efficient Implementation of Essentially Non-Oscillatory Shock-Capturing Schemes, II"*, Journal of Computational Physics, 83(1), pp. 32-78, 1989. Chapitres pertinents : [Chapitre 8](08_test_cases.md) (§8.7 — cas test Shu-Osher, interaction choc/onde entropique).

**[Sod, 1978]** G.A. Sod — *"A Survey of Several Finite Difference Methods for Systems of Nonlinear Hyperbolic Conservation Laws"*, Journal of Computational Physics, 27(1), pp. 1-31, 1978. Chapitres pertinents : [Chapitre 8](08_test_cases.md) (§8.1 — tube de Sod, cas test de reference pour la dynamique des gaz).

**[Sweby, 1984]** P.K. Sweby — *"High Resolution Schemes Using Flux Limiters"*, SIAM Journal on Numerical Analysis, 21(5), pp. 995-1011, 1984. Chapitres pertinents : [Chapitre 5](05_reconstruction.md) (§5.1 — region TVD de Sweby, conditions sur les limiteurs de pente).

**[Toro, 2009]** E.F. Toro — *"Riemann Solvers and Numerical Methods for Fluid Dynamics"*, 3e edition, Springer, 2009. Reference principale de ce projet. Chapitres pertinents : [Chapitre 1](01_euler_equations.md) (Ch. 1-3 de Toro — equations d'Euler, structure des ondes), [Chapitre 2](02_riemann_problem.md) (Ch. 4-5 de Toro — solution exacte du probleme de Riemann), [Chapitre 3](03_finite_volume.md) (Ch. 6 de Toro — methode de Godunov), [Chapitre 4](04_flux_schemes.md) (Ch. 10 de Toro — solveurs de Riemann approches), [Chapitre 6](06_time_integration.md) (Ch. 6 de Toro — integration temporelle), [Chapitre 7](07_fourier_analysis.md) (Ch. 13 de Toro — analyse de Fourier), [Chapitre 8](08_test_cases.md) (Ch. 6, 17 de Toro — cas test numeriques).

**[van Albada et al., 1982]** G.D. van Albada, B. van Leer, W.W. Roberts — *"A Comparative Study of Computational Methods in Cosmic Gas Dynamics"*, Astronomy & Astrophysics, 108, pp. 76-84, 1982. Chapitres pertinents : [Chapitre 5](05_reconstruction.md) (§5.1 — limiteur de pente van Albada, lisse et TVD).

**[van Leer, 1979]** B. van Leer — *"Towards the Ultimate Conservative Difference Scheme. V. A Second-Order Sequel to Godunov's Method"*, Journal of Computational Physics, 32, pp. 101-136, 1979. Chapitres pertinents : [Chapitre 5](05_reconstruction.md) (§5.1 — methode MUSCL, reconstruction lineaire par morceaux avec limiteur de pente).
