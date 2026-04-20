# <span style="color:blue"><b>TODO :</b></span> Classe `Variables`

- [x] Rajouter un evaluate général (en plus du `evaluate_quad`)
- [x] Mettre le `find_cell_index` dans le vmap dans un `classical_evaluate_2` (et comparer)
- [x] Adapter la projection/evaluation au cas `Cellwise`
- [x] Modifier `Variables` pour prendre en compte deux bases, la trial et la test (supposer que test est même type que trial ou analytique).

    | Trial | Test |
    |---------|--------|
    |`AnalyticBasis`| `AnalyticBasis`|
    |`PatchwiseParametricBasis`| `AnalyticBasis`|
    |`PatchwiseParametricBasis`| `PatchwiseParametricBasis`|
    |`CellwiseParametricBasis`| `AnalyticBasis`|
    |`CellwiseParametricBasis`| `CellwiseParametricBasis`|

- [x] Regarder la différence de temps et de convergence entre le cas `Patchwise` et `Cellwise` sur 5 cellules

- [x] Relancer tous les tests (+ passer les losses en ylog).

- [ ] Vérifier quelle signature est la meilleur pour le `_call__` de `Cellwise` :
    ```[python]
    def __call__(self, cell_module, i: int, inputs: jnp.ndarray) -> jnp.ndarray:
    def __call__(self, i: int, inputs: jnp.ndarray) -> jnp.ndarray:

    ```
