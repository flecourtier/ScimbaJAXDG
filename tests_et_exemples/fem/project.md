# <span style="color:blue"><b>Tests & Exemples :</b></span> Classe `VariablesFE` (project FEM)

## Project FEM - 1D

### Exemples sans apprentissage

- [x] **Exemple 1.a (`variable.py`) :** (TEST)

    - *Tester :* la classe `VariablesFE` sans post-processing (PP None) avec une base analytique (Taylor).

    - *Valider :* `classical_evaluation` et `classical_projection`.

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|
    | 10 | 2 | 3 | 5 | $1$ | / | $\varphi_{k,i}$ | / |
    | " | " | " | " | $x+1$ | " | " | " |
    | " | " | " | " | $2x^2 - 0.2x + 1$ | " | " | " |

- [x] **Exemple 1.b (`variable_mixed_basis.py`) :** (TEST)

    - *Tester :* la classe `VariablesFE` sans post-processing (PP None) avec une base mixte (analytique dans la première cellule et réseau dans la deuxième cellule).

- [x] **Exemple 2 (`variable_local_post_processing.py`) :** 

    - *Tester :* la classe `VariablesFE` avec post-processing (PP analytique) et une base analytique (Taylor).

    - *Détails :* Considérer un PP linéaire et non-linéaire. Valider l'utilisation du newton en linéaire (avec 1 itération).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP non apprenable).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|
    | 50 | 5 | 3 | 5 | $(x+1)^2$ | / | $\varphi_{k,i}$ | linéaire : $2w$ |
    | " | " | " | " | " | " | " | non-linéaire : $w^2$ |

- [ ] **Exemple 3 (`variable_local_post_processing_NN.py`) :** 

    - *Tester :* la classe `VariablesFE` avec post-processing (PP réseau) et une base analytique (Taylor).
    
    - *Détails :* Prendre $f(x) = sin(x^2+1)$, $5$ mailles et des bases d'ordre 2. Entrainer le réseau : $min_\theta \|P_{PP}(u)-f\|^2$ où $P_{PP}(u)$ est la projectionavec post-processing.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP apprenable).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    <!-- | 5 | 4 | 2 | 1 | $\sin(x^2+1)$ | / | $\varphi_{k,i}$ | $\mathcal{P}_\theta$ | [(voir)](images/project/variable_local_post_processing_NN.png) | | -->


- [x] **Exemple 4.a (`variable_patchwise_basis.py`) :** 

    - *Tester :* la classe `VariablesFE` sans post-processing et une base réseau (`Patchwise`).

    - *Détails :* Ne pas utiliser les points de quadrature pour l'entrainement du réseau.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/project/variable_patchwise_basis.png) |

- [x] **Exemple 4.b (`variable_patchwise_basis_with_mapping.py`) :** 

    - *Tester :* la même chose. Ajouter un `Mapping` entraînable pour le maillage.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour mapping apprenable).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 12 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | $g_\theta$ | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/project/variable_patchwise_basis_with_mapping.png) |

- [x] **Exemple 5 (`variable_patchwise_basis_pretrained.py`) :** 

    - *Tester :* la classe `VariablesFE` sans post-processing et une base réseau pré-entraîné (`Patchwise`).

    - *Détails :* Considérer $\phi_i^\theta(x) = v_\theta(x) \phi_i(x)$, où $v_\theta$ est un réseau de neurones (ex: MLP) pré-entrainés et $\phi_i$ une base analytique (Taylor).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/project/variable_patchwise_basis_pretrained.png) |

### Exemples avec apprentissage (via classe `DGProjectionApproximationSpace`)

- [x] **Exemple 1 (`dgprojection_patchwise_basis_with_mapping.py`) :** [(voir)](images/project/dgprojection_patchwise_basis_with_mapping.png)

    - *Tester :* la classe `DGProjectionApproximationSpace` pour faire une projection avec une base `Patchwise` et un mapping entraînable. -->