# <span style="color:blue"><b>Tests & Exemples :</b></span> Classe `PhiFEMDirichlet` (solve $\varphi$-FEM)

## Solve $\varphi$-FEM - 2D (Exemples sans apprentissage)

- [x] **Exemple 1.a (`classify_2d_cells_and_facets_circle.py`) :** 
      [(voir)](classify_2d_cells_and_facets_circle.png)

    - *Tester :* la classe `LevelSetClassifier` pour la sélection de cellule et de faces pour le schéma $\varphi$-FEM.

- [ ] **Exemple 1.b (`solve_2d_dirichlet_circle.py`) :** 
      [(voir)](solve_2d_dirichlet_circle.png)

    - *Tester :* la classe `PhiFEMDirichlet` pour la résolution du problème de Poisson avec $\varphi$-FEM sur un cercle. 
    - *Valider :* les courbes de convergence en $L^2$, $L^\infty$ et $H^1$.