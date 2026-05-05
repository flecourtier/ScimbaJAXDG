# <span style="color:blue"><b>Tests & Exemples :</b></span> Classe `Variables` (project DG)

## Project DG - 1D

### Exemples sans apprentissage

- [x] **Exemple 1.a (`variable.py`) :** (TEST)

    - *Tester :* la classe `Variables` sans post-processing (PP None) avec une base analytique (Taylor).

    - *Valider :* `classical_evaluation` et `classical_projection`.

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|
    | 10 | 2 | 3 | 5 | $1$ | / | $\varphi_{k,i}$ | / |
    | " | " | " | " | $x+1$ | " | " | " |
    | " | " | " | " | $2x^2 - 0.2x + 1$ | " | " | " |


- [x] **Exemple 1.b (`variable_mixed_basis.py`) :** (TEST)

    - *Tester :* la classe `Variables` sans post-processing (PP None) avec une base mixte (analytique dans la première cellule et réseau dans la deuxième cellule).

- [x] **Exemple 2 (`variable_local_post_processing.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP analytique) et une base analytique (Taylor).

    - *Détails :* Considérer un PP linéaire et non-linéaire. Valider l'utilisation du newton en linéaire (avec 1 itération).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP non apprenable).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|
    | 200 | 4 | 3 | 5 | $(x+1)^2$ | / | $\varphi_{k,i}$ | linéaire : $2w$ |
    | " | " | " | " | " | " | " | non-linéaire : $w^2$ |

- [x] **Exemple 3 (`variable_local_post_processing_NN.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP réseau) et une base analytique (Taylor).
    
    - *Détails :* Prendre $f(x) = sin(x^2+1)$, $5$ mailles et des bases d'ordre 2. Entrainer le réseau : $min_\theta \|P_{PP}(u)-f\|^2$ où $P_{PP}(u)$ est la projectionavec post-processing.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP apprenable).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 5 | 4 | 2 | 1 | $\sin(x^2+1)$ | / | $\varphi_{k,i}$ | $\mathcal{P}_\theta$ | [(voir)](images/project/variable_local_post_processing_NN.png) |

- [x] **Exemple 4.a (`variable_patchwise_basis.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau (`Patchwise`).

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
    | 1 | 10 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | $g_\theta$ | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/project/variable_patchwise_basis_with_mapping.png) |

- [x] **Exemple 5 (`variable_patchwise_basis_pretrained.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau pré-entraîné (`Patchwise`).

    - *Détails :* Considérer $\phi_i^\theta(x) = v_\theta(x) \phi_i(x)$, où $v_\theta$ est un réseau de neurones (ex: MLP) pré-entrainés et $\phi_i$ une base analytique (Taylor).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|--------|
    | 1 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,P}$ | / | [(voir)](images/project/variable_patchwise_basis_pretrained.png) |


- [x] **Exemple 6 (`variable_cellwise_basis.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau (`Cellwise`).

    - *Détails :* Même exemple que exemple 4.a mais un réseau par maille. (On pourra essayer de rajouter une condition de continuité entre les mailles dans la loss pour voir l'impact).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Cellwise`).

    ---

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|-------------|
    | 2 | 20 | 1 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,C}$ | / | [(voir)](images/project/variable_cellwise_basis.png),[(zoom)](images/project/variable_cellwise_basis_zoom.png) |
    | 5 | " | " | " | " | " | " | " | [(voir)](images/project/variable_cellwise_basis_nc5.png),[(zoom)](images/project/variable_cellwise_basis_nc5_zoom.png) |
    | 10 | " | 2 | " | " | " | " | " | [(voir)](images/project/variable_cellwise_basis_nc10.png) |

- [x] **Exemple 7 (`variable_cellwise_basis_and_local_post_processing_NN.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP réseau) et une base réseau (`Cellwise`).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP apprenable et bases apprenable - `Cellwise`).

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|-------------|
    | 8 | 20 | 2 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,C}$ | $\mathcal{P}_\theta$ | [(voir)](images/project/variable_cellwise_basis_and_local_post_processing_NN.png),[(zoom)](images/project/variable_cellwise_basis_and_local_post_processing_NN_zoom.png) |

- [x] **Exemple 8 (`compare_patchwise_cellwise.py`) :**

    - *Détails :* Comparer les cas `Patchwise` et `Cellwise` sur 5 cellules, 20 points de quadrature, des bases d'ordre 2 et la même fonction à projeter. Ne pas utiliser de post-processing ni de mapping.

    | $n_\text{cells}$ | $n_\text{quad}$ | $n_b$ | $n_\text{out}$ | $f(x)$ | Mapping | Bases | Post-processing | |
    |---------|--------|-------------|---------|-------------|---------|--------|-------------|-------------|
    | 5 | 20 | 2 | 1 | $\sin(2\pi(x^2+1)) + 2$ | / | $\varphi_{k,i}^{\theta,P}$/$\varphi_{k,i}^{\theta,C}$ | / | [(voir)](images/project/compare_patchwise_cellwise.png) |

### Exemples avec apprentissage (via classe `DGProjectionApproximationSpace`)

- [x] **Exemple 1 (`dgprojection_patchwise_basis_with_mapping.py`) :** [(voir)](images/project/dgprojection_patchwise_basis_with_mapping.png)

    - *Tester :* la classe `DGProjectionApproximationSpace` pour faire une projection avec une base `Patchwise` et un mapping entraînable.