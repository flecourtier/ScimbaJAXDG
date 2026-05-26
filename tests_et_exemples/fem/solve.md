# <span style="color:blue"><b>Tests & Exemples :</b></span> Classe `EllipticFEscheme` (solve FEM)

## Solve FEM - 1D

### Exemples sans apprentissage

#### Linéaire

- [x] **Exemple 1 (`solve_1d_laplacian.py`) :** [(voir)](images/solve/solve_1d_laplacian.png)

    - *Tester :* la classe `EllipticFEscheme` pour la résolution de l'équation de Poisson (Laplacien) avec une base analytique (Lagrange).
    - *Valider :* les courbes de convergence en $L^2$, $L^\infty$ et $H^1$.