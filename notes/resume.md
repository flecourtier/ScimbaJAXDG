# Résumé - Code DG

Voici un résumé fichier par fichier du code DG de Scimba  (dans src/scimba_jax).

Les fichiers de bases sont dans "linear_approximation" :

- **quad/gauss_quad.py :** construit les points/poids de quadrature volumiques et surfaciques sur l’élément de référence.
- **meshes/mesh.py :** gère tout le partie du maillage (à dimension spatiale quelconque).
- **(meshes/mesh_1d.py** : version 1D simplifiée (plus lisible pour démarrer DG en 1D). )
- **basis/general_bases.py :** définit les trois types de bases possible - analytique, patchwise (même réseau sur l'ensemble des "patch"/cellules), cellwise (un réseau par "patch").
- **analytic_bases.py : implémentations** concrètes de bases analytiques - Taylor (pour le DG) et Lagrange (pour les EF).
- **variables/postprocessing.py :** post-traitement local des variables (linéaire ou non, potentiellement apprenable).
- **variables/variables_dg.py :** gère la construction de la variable DG - regroupe les bases (tests/trials) et les dofs + évaluation locale/globale + projection sur la base.
- **dg/elliptic_dg_scheme.py :** assemblage complet du résidu DG elliptique (via les termes de volume, les flux internes et les flux bord) et résolution Newton.
- **dg/flux.py :** flux numériques DG (interface et bord) avec 3 variantes SIPG/NIPG/Babuška-Zlámal.
- **dg/solve.py :** solveurs Newton (classique et matrix-free) avec différentiation implicite.

Quelques fichiers supplémentaires :

- **mapping/mapping.py :** gère les transformations géométriques via la composition de mapping (inversible) qui sont potentiellement apprenables.
- **model_class/funcparam_vectorial.py :** définit les wrappers ParamFunc (scalaire, vectoriel, champ) et les opérateurs de différentiabilité (gradient, jacobien, laplacien, etc.).
- **physical_models/abstract_linear_weak_form.py :** interface PDE linéaire en forme faible (bilinear_form, linear_form).
- **physical_models/classical_weakform/laplacian_weak_form.py** : cas concret du Laplacien.
- **nonlinear_approximation/approximation_spaces/dg_approximation_spaces.py :** dernier fichier un peu "technique" qui permet de définir des espaces d’approximation DG de "haut niveau". C’est ce fichier qui permet d’intégrer ce qui est fait en DG dans les modules de résolution généraux de Scimba (par exemple avec les PINNs).