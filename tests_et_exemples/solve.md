# <span style="color:blue"><b>Tests & Exemples :</b></span> Classe `EllipticDGscheme` (solve DG)

## Solve DG - 1D

### Exemple sans apprentissage

- [x] **Exemple 1.a (`solve_laplacian.py`) :** [(voir)](images/solve/solve_laplacian.png)

    - *Tester :* la classe `EllipticDGscheme` pour la résolution de l'équation de Poisson (Laplacien) avec une base analytique (Taylor).

    - *Valider :* les courbes de convergence en $L^2$, $L^\infty$ et $H^1$ pour les flux SIPG, NIPG et BZ.

- [x] **Exemple 1.b (`solve_laplacian_matrix_free.py`) :** [(voir)](images/solve/solve_laplacian_matrix_free.png)

    - *Tester :* la même chose mais en version "matrix free".

    - *Valider :* les courbes de convergence en $L^2$, $L^\infty$ et $H^1$ pour les flux SIPG et BZ.

- [x] **Exemple 2 (`solve_diffusion.py`) :** [(voir)](images/solve/solve_diffusion.png)

    - *Tester :* la classe `EllipticDGscheme` pour la résolution de l'équation de diffusion avec une base analytique (Taylor).

- [x] **Exemple 3 (`solve_diffusion_advection.py`) :** [(voir)](images/solve/solve_diffusion_advection.png)

    - *Tester :* la classe `EllipticDGscheme` pour la résolution de l'équation de diffusion/advection avec une base analytique (Taylor).
  
- [x] **Exemple 4 (`solve_diffusion_advection_reaction.py`) :** [(voir)](images/solve/solve_diffusion_advection_reaction.png)

    - *Tester :* la classe `EllipticDGscheme` pour la résolution de l'équation de diffusion/advection/réaction avec une base analytique (Taylor).

### Exemple avec apprentissage (via classe `DGEllipticApproximationSpace`)

- [x] **Exemple 1.a (`dgelliptic_patchwise_basis.py`) :** [(voir)](images/solve/dgelliptic_patchwise_basis.png)

    - *Tester :* la classe `DGEllipticApproximationSpace` pour la résolution de l'équation de Poisson (Laplacien) avec une base réseau (`Patchwise`).

- [ ] **Exemple 1.b (`dgelliptic_patchwise_basis_matrix_free.py`) :**

    - *Tester :* la même chose mais en version "matrix free".

- [x] **Exemple 2 (`dgelliptic_patchwise_basis_with_mapping.py`) :** [(voir)](images/solve/dgelliptic_patchwise_basis_with_mapping.png)

    - *Tester :* la même chose. Ajouter un `Mapping` entraînable pour le maillage.

- [x] **Exemple 3 (`dgelliptic_cellwise_basis.py`) :** [(voir)](images/solve/dgelliptic_cellwise_basis.png)

    - *Tester :* la classe `DGEllipticApproximationSpace` pour la résolution de l'équation de Poisson (Laplacien) avec une base réseau (`Cellwise`).


## Solve DG - 2D

- [x] **Exemple 1 (`solve_laplacian_2d.py`) :** [(voir - cvg)](images/solve/solve_laplacian_2d.png) [(voir - plot)](images/solve/solve_laplacian_2d_plot.png)

    - *Tester :* la classe `EllipticDGscheme` pour la résolution de l'équation de Poisson (Laplacien) en 2D avec une base analytique (Taylor). 

    - *Valider :* les courbes de convergence en $L^2$, $L^\infty$ et $H^1$ pour SIPG.