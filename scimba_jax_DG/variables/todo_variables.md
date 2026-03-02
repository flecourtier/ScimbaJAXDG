# <span style="color:blue"><b>TODO :</b></span> Classe `Variables`

## **Notations**

**Généralités :**

- $d=1$ : dimension physique
- $n_c$ : nombre de cellules
- $n_{\text{quad}}$ : nombre de points de quadrature
- $n_b$ : ordre de la base (`order`)
- $n_u$ : nombre total de variables (`out_dim`)
- $f$ : fonction à projeter

**Spécifique à ScimBa :** (mapping, post-processing et/ou bases apprenables)

- $\theta$ : paramètres du réseau de neurones (notation générique pour le mapping, le post-processing et les bases apprenables)
- $g, g_\theta$ : mapping du maillage (composition de $m$ fonctions) $\longrightarrow$ analytique, réseau de neurones
- $\mathcal{P}, \mathcal{P}_\theta$ : post-processing $\longrightarrow$ analytique, réseau de neurones
- $\varphi_{k,i}$, $\varphi_{k,i}^{\theta,P}$, $\varphi_{k,i}^{\theta,C}$ : la $i$-ème fonction de base (trial) dans la $k$-ème cellule $\longrightarrow$ Taylor, un réseau de neurones (`Patchwise`), $n_c$ réseaux de neurones (`Cellwise`)

## **Cas tests**

- [x] **Cas test 1 (`variable.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing (PP None) avec une base analytique (Taylor).

    - *Valider :* `classical_evaluation` et `classical_projection`.

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|
    | 10 | 2 | 3 | 5 | $1$ | / | $\varphi_{k,i}$ | / |
    | " | " | " | " | $x+1$ | " | " | " |
    | " | " | " | " | $2x^2 - 0.2x + 1$ | " | " | " |

- [x] **Cas test 2 (`variable_local_post_processing.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP analytique) et une base analytique (Taylor).

    - *Détails :* Considérer un PP linéaire et non-linéaire. Valider l'utilisation du newton en linéaire (avec 1 itération).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP non apprenable).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|
    | 200 | 4 | 3 | 5 | $(x+1)^2$ | / | $\varphi_{k,i}$ | linéaire : $2w$ |
    | " | " | " | " | " | " | " | non-linéaire : $w^2$ |

- [x] **Cas test 3 (`variable_local_post_processing_NN.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP réseau) et une base analytique (Taylor).
    
    - *Détails :* Prendre $f(x) = sin(x^2+1)$, $5$ mailles et des bases d'ordre 2. Entrainer le réseau : $min_\theta \|P_{PP}(u)-f\|^2$ où $P_{PP}(u)$ est la projectionavec post-processing.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP apprenable).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 5 | 4 | 2 | 1 | $\sin(x^2+1)$ | / | $\varphi_{k,i}$ | $\mathcal{P}_\theta$ | [(voir)](images/variable_local_post_processing_NN.png) |

- [x] **Cas test 4.a (`variable_patchwise_basis.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau (`Patchwise`).

    - *Détails :* Ne pas utiliser les points de quadrature pour l'entrainement du réseau.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/variable_patchwise_basis.png) |


- [x] **Cas test 4.b (`variable_patchwise_basis_with_mapping.py`) :** 

    - *Tester :* la même chose. Ajouter un `Mapping` entraînable pour le maillage.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour mapping apprenable).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 10 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | $g_\theta$ | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/variable_patchwise_basis_with_mapping.png) |

- [x] **Cas test 5 (`variable_patchwise_basis_pretrained.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau pré-entraîné (`Patchwise`).

    - *Détails :* Considérer $\phi_i^\theta(x) = v_\theta(x) \phi_i(x)$, où $v_\theta$ est un réseau de neurones (ex: MLP) pré-entrainés et $\phi_i$ une base analytique (Taylor).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/variable_patchwise_basis_pretrained.png) |


- [x] **Cas test 6 (`variable_cellwise_basis.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau (`Cellwise`).

    - *Détails :* Même cas test que cas test 4.a mais un réseau par maille. (On pourra essayer de rajouter une condition de continuité entre les mailles dans la loss pour voir l'impact).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Cellwise`).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|-------------|
    | 2 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,C}$ | / | [(voir)](images/variable_cellwise_basis.png),[(zoom)](images/variable_cellwise_basis_zoom.png) |
    | 5 | " | " | " | " | " | " | " | [(voir)](images/variable_cellwise_basis_nc5.png),[(zoom)](images/variable_cellwise_basis_nc5_zoom.png) |
    | 10 | " | 2 | " | " | " | " | " | [(voir)](images/variable_cellwise_basis_nc10.png) |

- [x] **Cas test 7 (`variable_cellwise_basis_and_local_post_processing_NN.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP réseau) et une base réseau (`Cellwise`).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP apprenable et bases apprenable - `Cellwise`).

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|-------------|
    | 8 | 20 | 2 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,C}$ | $\mathcal{P}_\theta$ | [(voir)](images/variable_cellwise_basis_and_local_post_processing_NN.png),[(zoom)](images/variable_cellwise_basis_and_local_post_processing_NN_zoom.png) |

- [ ] **Cas test 8 (...) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base analytique (Taylor), mais avec un `Mapping` entrainable.

    - *Détails :* ...

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour mapping apprenable).

- [ ] **Supplémentaires :**

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

    - [ ] Vérifier quelle signature est la meilleur pour le `_call__` de `Cellwise` :
        ```[python]
        def __call__(self, cell_module, i: int, inputs: jnp.ndarray) -> jnp.ndarray:
        def __call__(self, i: int, inputs: jnp.ndarray) -> jnp.ndarray:
        ```
    - [ ] Regarder la différence de temps et de convergence entre le cas `Patchwise` et `Cellwise` sur 5 cellules