# <span style="color:blue"><b>TODO :</b></span> Classe `Variables`

- [x] **Cas test 1 (`variable.py`) :** 

    - *Tester :* la classe `Variables` sans post-processing (PP None) avec une base analytique (Taylor).

    - *Valider :* `classical_evaluation` et `classical_projection`.


- [x] **Cas test 2 (`variable_local_post_processing.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP analytique) et une base analytique (Taylor).

    - *Détails :* Considérer un PP linéaire et non-linéaire. Valider l'utilisation du newton en linéaire (avec 1 itération).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP non apprenable).

- [x] **Cas test 3 (`variable_local_post_processing_NN.py`) :** 

    - *Tester :* la classe `Variables` avec post-processing (PP réseau) et une base analytique (Taylor).
    
    - *Détails :* Prendre $f(x) = sin(x^2+1)$, $5$ mailles et des bases d'ordre 2. Entrainer le réseau : $min_\theta \|P_{PP}(u)-f\|^2$ où $P_{PP}(u)$ est la projectionavec post-processing.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour un PP apprenable).

- [ ] **Cas test 4 (...) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau (`Patchwise`).

    - *Détails :* Ne pas utiliser les points de quadrature pour l'entrainement du réseau.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

    <!-- Prendre $f(x) = (x^2+2x+1)sin(2x)$. -->

- [ ] **Cas test 5 (...) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau pré-entraîné (`Patchwise`).

    - *Détails :* Considérer $\phi_i^\theta(x) = v_\theta(x) \phi_i(x)$, où $v_\theta$ est un réseau de neurones (ex: MLP) pré-entrainés et $\phi_i$ une base analytique (Taylor).

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Patchwise`).

- [ ] **Cas test 6 (...) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base réseau (`Cellwise`).

    - *Détails :* Même cas test que cas test 4 mais un réseau par maille.

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour bases apprenable - `Cellwise`).

- [ ] **Cas test 7 (...) :** 

    - *Tester :* la classe `Variables` sans post-processing et une base analytique (Taylor), mais avec un `Mapping` entrainable.

    - *Détails :* ...

    - *Valider :* `local_postprocessing_evaluate` et `projector_with_nonlinearlocal_postprocessing` (pour mapping apprenable).

- [ ] **Supplémentaires :**

    - [ ] Rajouter un evaluate général (en plus du `evaluate_quad`)
    - [ ] Adapter la projection/evaluation au cas `Cellwise`
    - [ ] Modifier `Variables` pour prendre en compte deux bases, la trial et la test (supposer que test est même type que trial ou analytique).
