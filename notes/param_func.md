
# Fonctions paramétrées (`ParamFunc`)

> src/scimba_jax/nonlinear_approximation/model_class/funcparam.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_scalar.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_field.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_vectorial.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_matrix.py

Les `ParamFunc` sont des **wrappers autour de fonctions** de la forme :

$$f : (\theta, x) \longmapsto y$$

où $\theta$ est un pytree (e.g. paramètres d'un réseau de neurones ou `scheme_pytree`), et $x \in \mathbb{R}^d$ (avec $d$ la dimension spatiale). Elles permettent d'appliquer des opérateurs différentiels (gradient, laplacien, divergence...) via `jax.grad` / `jax.jacfwd`.

## Les quatre types

| Classe | Sortie $y$ | Taille | Attributs clés |
|---|---|---|---|
| `ParamScalarFunction` | scalaire | $(1,)$ | `dim`, `fn`, `f_type` |
| `ParamFieldFunction` | vecteur de taille $d$ | $(d,)$ | `dim`, `fn`, `f_type` |
| `ParamVecFunction` | vecteur de taille $s$ quelconque | $(s,)$ | `size`, `dim`, `fn`, `f_type` |
| `ParamMatrixFunction` | matrice $(d \times d)$ | $(d, d)$ | `dim`, `fn`, `f_type` |

> **Remarque :** `ParamFieldFunction` est un cas particulier de `ParamVecFunction` où `size = dim` — représente un champ vectoriel (e.g. $\nabla u$). `ParamMatrixFunction` est typiquement utilisée pour les jacobiens et hessiens.

## `f_type` — Type de signature

Le `f_type` encode les arguments de $f$ sous forme de chaîne de caractères :

| `f_type` | Signature | Usage typique |
|---|---|---|
| `"x"` | $(\theta, x)$ | cas DG (scheme pytree + point) |
| `"x_mu"` | $(\theta, x, \mu)$ | solution paramétrée en $\mu$ |
| `"t_x_mu"` | $(\theta, t, x, \mu)$ | solution dépendante du temps |

## Opérateurs disponibles

Les opérateurs sont composables et retournent une nouvelle `ParamFunc` du même type ou d'un type réduit :

| Opérateur | Sur | Retourne | Description |
|---|---|---|---|
| `+`, `-`, `*`, `/` | toutes | même type | algèbre pointwise |
| `gradient_x()` | `ParamScalarFunction` | `ParamFieldFunction` | $\nabla_x f$ |
| `partial_derivative_x(i)` | `ParamScalarFunction` | `ParamScalarFunction` | $\partial_{x_i} f$ |
| `laplacian_x()` | `ParamScalarFunction` | `ParamScalarFunction` | $\Delta_x f = \mathrm{tr}(H_x f)$ |
| `hessian_x()` | `ParamScalarFunction` | `ParamMatrixFunction` | $H_x f$ |
| `divergence_x()` | `ParamFieldFunction` | `ParamScalarFunction` | $\nabla \cdot f$ |
| `jacobian_x()` | `ParamFieldFunction` | `ParamMatrixFunction` | $J_x f$ |
| `dot(other)` | `ParamFieldFunction` | `ParamScalarFunction` | $f \cdot g$ |
| `component(i)` | toutes | `ParamScalarFunction` | $f_i$ |
| `compose(g)` | toutes | même type | $f(\theta, g(\cdot))$ |
