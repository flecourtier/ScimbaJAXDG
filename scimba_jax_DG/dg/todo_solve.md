# <span style="color:blue"><b>TODO :</b></span> Classe `EllipticDGscheme` - Résolution

- [x] Modifier `__call__`des flux pour retourner un seul élément (ancienne version à reprendre)
- [x] Implémenter le flux SIPG (référence : [Unified Analysis of Discontinuous Galerkin Methods for Elliptic Problems. Douglas N. Arnold, Franco Brezzi, Bernardo Cockburn, and L. Donatella Marini](https://www-users.cse.umn.edu/~arnold/papers/dgerr.pdf)) et tester l'assemblage du flux sur un problème simple (dofs à 1)
- [x] Ajouter les notes sur le flux SIPG en présentant l'exemple.
- [x] Ajouter les conditions de Dirichlet et tester à nouveau l'assemblage du flux
- [x] Modifier le calcul de l'erreur $L^2$ pour qu'elle soit calculée maille par maille (et non pas globalement)
- [x] Tester le problème de Poisson avec solution polynomiale d'ordre 2 (avec $\Rightarrow f=1$) et comparer la solution numérique à la solution exacte $\Rightarrow$ doit être exact pour $k=2$
- [x] Tester Poisson avec $f$ sinusoïdal et comparer la solution numérique à la solution exacte en vérifiant les convergences en $L^2$ et $L^\infty$
- [ ] Tester avec des termes sources différents
- [ ] Tester sur un problème élliptique plus complexe (ajouter une matrice de diffusion)
- [ ] Regarder les perfs : pourquoi ça mets autant de temps à tourner ?

- [ ] **WARNING !!!** Réparer le flux centré.