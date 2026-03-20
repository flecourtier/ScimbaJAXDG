# Implémentation : Branche `scimba_jax_dg`

---

On considère la dimension physique $d=1$.

## 1. Mapping 📖

> src/scimba_jax/mapping/mapping.py

* `@dataclass(frozen=True)` $\rightarrow$ génère `__init__`, `__str__`, `__eq__` $\rightarrow$ Instance immuable.
* **Mapping** = Composition de `InvertibleFunction` et `InvertibleModule`.

$$g = g_m \circ g_{m-1} \circ ... \circ g_1$$

avec $m$ le nombre de fonctions composées. Peut dépendre de $\theta$ (paramètres d'un réseau de neurones).

## 2. Quadrature 📖

> src/scimba_jax/linear_approximation/quad/gauss_quad.py

$$ \int_{0}^1 f(x) dx = \sum_{i=1}^{n_{\text{quad}}} w_i f(x_i) $$

où $n_{\text{quad}}$ est le degré/nombre de points de quadrature (`ordre` = `n_points` pour $d=1$).

**Dimensions des tenseurs :**
* `points_1D`,`weights_1D` : $(n_{\text{quad}},)$, $(n_{\text{quad}},)$
* `volumic_points`, `volumic_weights` : $(n_{\text{quad}}^d, d)$, $(n_{\text{quad}}^d,)$
* `surfacic_points` : liste de taille $2d$ (2 faces par dim) où chaque élément est $(n_{\text{quad}}^{d-1}, d)$

    `surfacic_weights` : liste de taille $2d$ (2 faces par dim) où chaque élément est $(n_{\text{quad}}^{d-1},)$

## 3. Maillage

> src/scimba_jax/linear_approximation/meshes/mesh_1d.py

Passage sur l'élément de référence pour l'intégration :

$$ \int_{\Omega} f(X) dX = \int_{\Omega_{\text{ref}}} f(\Phi(\xi)) |\det J_{\Phi}(\xi)| d\xi $$

Discrétisation :

$$ \int_{\Omega} f(X) dX  \approx \sum w_i f(X_i) $$

$$ \int_{\Omega_{\text{ref}}} f(\Phi(\xi)) |\det J_{\Phi}(\xi)| d\xi \approx \sum |\det J_{\Phi}(\xi_i)| w_i f(\Phi(\xi_i)) $$

## 4. Bases

> src/scimba_jax/linear_approximation/basis/general_bases.py
> src/scimba_jax/linear_approximation/basis/analytic_bases.py

On note :

- $d$ : la dimension physique (pour l'instant $d=1$)
- $n_c$ : le nombre de cellules
- $n_{\text{quad}}$ : le nombre de points de quadrature
- $n_b$ : l'ordre de la base (`order`)
- $n_u$ :  le nombre total de variables (`out_dim`)

Chaque variable peut s'écrire sous la forme :

$$ u(x) = \sum_{k=0}^{n_c-1} \sum_{i=0}^{n_b-1} u_{k,i} \bar{\varphi}_{k,i}(x) $$

où $\bar{\varphi}_{k,i}(x)$ est la $i$-ème fonction de base (trial) dans la $k$-ème cellule. On notera $\varphi_{k,i}(x)$ pour la base analytique (Taylor) et $\varphi_{k,i}^{\theta,P}(x)$, $\varphi_{k,i}^{\theta,C}(x)$ pour les bases paramétrées (Patchwise, Cellwise).

### Analytic Basis (Taylor)
$$ \varphi_{k,i}(x) = \frac{(x - x_k)^i}{i!} \quad \longrightarrow \quad (n_b, n_u). $$
où $x_k$ est le centre de la cellule $k$, 

**Dimensions :**
* $x$ : pts de quadrature (global) $\quad \longrightarrow \quad (n_c, n_{\text{quad}}, d)$
* $y = \{\varphi_{k,i}(x)\}$ : évaluation des fonctions de bases au pts de quadrature $\quad \longrightarrow (n_c, n_{\text{quad}}, n_b, n_u)$
* $\partial_x y \quad \longrightarrow (n_c, n_{\text{quad}}, n_b, n_u, d)$

### Bases paramétrées (Machine Learning)
* **Patchwise Parametric Basis :** 
  $$ \varphi_{k,i}^{\theta,P}(x) = v_\theta(x) \varphi_{k,i}(x) $$
* **Cellwise Parametric Basis :**
  $$ \varphi_{k,i}^{\theta,C}(x) = v_{k,\theta}(x) \varphi_{k,i}(x) $$

## 5. Post-processing

> src/scimba_jax/linear_approximation/variables/postprocessing.py

- `LocalPostProcessing` : post-processing local sur chaque cellule
- Peut être linéaire ou non, peut-être analytique ou réseau de neurones

## 6. Variables

> src/scimba_jax/linear_approximation/variables/variables.py

$$ u_h(x) = \sum_{k=0}^{n_c-1} \sum_{i=0}^{n_b-1} u_{k,i} \bar{\varphi}_{k,i}(x) \mathbb{1}_{\{x \in C_k\}} $$

### Évaluation locale

$x \in C_k$, avec $k \in [\![0, n_c-1]\!]$.
$$ u_h(x) = \sum_{i=0}^{n_b-1} u_{k,i} \bar{\varphi}_{k,i}(x) $$

### Projection de $f$ sur l'espace DG
$\hookrightarrow$ Chercher les $u_{k,i}$ tels que $u_h$ soit la fonction approchée de $f$.

On considère les mêmes bases trial et test.

Pour $C_k$ :
$$ \int_{C_k} (f - u_h) \bar{\varphi}_{k,j} = 0 \quad \forall j $$

$$ \underbrace{\int_{C_k} f \cdot \bar{\varphi}_{k,j}}_{b_j} = \sum_{i=0}^{n_b-1} u_{k,i} \underbrace{\int_{C_k} \bar{\varphi}_{k,i} \cdot \bar{\varphi}_{k,j}}_{M_{i,j}} $$

$\hookrightarrow$ **Projecteur local :** $$ M_k U_k = b_k $$

**Dimensions :**

- `x = m.evaluate_mesh_points()` : pts de quadrature (global) $\quad \longrightarrow \quad (n_c, n_{\text{quad}}, d)$
- `variables.evaluate_quad(x)` : évaluation des variables sur les pts de quadrature $\quad \longrightarrow \quad (n_c, n_{\text{quad}}, n_u)$.

## 7. Fonctions paramétrées (`ParamFunc`)

> src/scimba_jax/nonlinear_approximation/model_class/funcparam_scalar.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_field.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_vectorial.py
> src/scimba_jax/nonlinear_approximation/model_class/funcparam_matrix.py

Les `ParamFunc` sont des **wrappers autour de fonctions** de la forme :

$$f : (\theta, x) \longmapsto y$$

où $\theta$ est un pytree (e.g. paramètres d'un réseau de neurones ou `scheme_pytree`), et $x \in \mathbb{R}^d$. Elles permettent d'appliquer des opérateurs différentiels (gradient, laplacien, divergence...) via `jax.grad` / `jax.jacfwd`.

### Les quatre types

| Classe | Sortie $y$ | Taille | Attributs clés |
|---|---|---|---|
| `ParamScalarFunction` | scalaire | $(1,)$ | `dim`, `fn`, `f_type` |
| `ParamFieldFunction` | vecteur de taille $d$ | $(d,)$ | `dim`, `fn`, `f_type` |
| `ParamVecFunction` | vecteur de taille $s$ quelconque | $(s,)$ | `size`, `dim`, `fn`, `f_type` |
| `ParamMatrixFunction` | matrice $(d \times d)$ | $(d, d)$ | `dim`, `fn`, `f_type` |

> **Remarque :** `ParamFieldFunction` est un cas particulier où `size = dim` — représente un champ vectoriel (e.g. $\nabla u$). `ParamMatrixFunction` est typiquement utilisée pour les jacobiens et hessiens.

### `f_type` — Type de signature

Le `f_type` encode les arguments de $f$ sous forme de chaîne de caractères :

| `f_type` | Signature | Usage typique |
|---|---|---|
| `"x"` | $(\theta, x)$ | cas DG (scheme pytree + point) |
| `"x_mu"` | $(\theta, x, \mu)$ | solution paramétrée en $\mu$ |
| `"t_x_mu"` | $(\theta, t, x, \mu)$ | solution dépendante du temps |

### Opérateurs disponibles

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

## 8. Équations différentielles - linéaire, forme faible (`AbstractLinearWeakForm`)

> src/scimba_jax/physical_models/abstract_linear_weak_form.py

Classe abstraite dont héritent tous les PDEs linéaires du schéma DG. Elle définit l'interface que l'assembleur `AbstractEllipticDGscheme` attend :
| Méthode / attribut | Rôle |
|---|---|
| A - `fields` | dictionnaire de fonctions non-apprenables (ex: `{"f": lambda x: ...}`) |
| A - `models` | liste des modules apprenables (`eqx.Module`) — e.g. réseaux de neurones |
| M - `bilinear_form(u, v)` | à implémenter — retourne l'intégrande $B(u,v)(x)$ comme `ParamFunc` |
| M - `linear_form(v)` | à implémenter — retourne l'intégrande $L(v)(x)$ comme `ParamFunc` |
| M - `list_learnable_fields()` | retourne les champs apprenables wrappés en `ParamFunc` |

### Le Laplacien (`LaplacianWeakForm`)  

> src/scimba_jax/physical_models/classical_weakform/laplacian_weak_form.py

Pour le problème $-\Delta u = f$ sur $\Omega$, la formulation faible donne :
$$\int_\Omega \nabla u \cdot \nabla v = \int_\Omega f\, v \quad \forall v$$
`LaplacianWeakForm` hérite de `AbstractLinearWeakForm` et implémente les deux formes comme des opérations sur les `ParamFunc` (voir section 7) :
```python
def bilinear_form(u, v):
    return u.gradient_x().dot(v.gradient_x())  # ∇u · ∇v  (intégrande)
def linear_form(v):
    return self.fields["f"] * v                # fv  (intégrande)
```

## 9. Schéma DG elliptique (`EllipticDGscheme`)

> src/scimba_jax/linear_approximation/dg/elliptic_dg_scheme.py

### Terme de flux

> src/scimba_jax/linear_approximation/dg/flux.py

**Interface numérique (`AbstractFlux`) :**

Considérons $n_f=n_c+1$ le nombre de faces et $l \in \{1, ..., n_f-2\}$ (`idf`) un indice de face intérieure.

On définit $C_L$ et $C_R$ comme les deux cellules voisines (gauche et droite) de la face $F_l$ .

Le flux numérique est défini par la classe abstraite `AbstractFlux` via `__call__(varL, varR)` où `varL` (resp. `varR`) est définit par :

$$\text{var}_L = (u_L,\ v_L,\ \nabla u_L,\ \nabla v_L,\ n_L,\ \text{fields}_L)$$

où $u_L$ (resp. $u_R$) est la solution dans la cellule $C_L$ (resp. $C_R$), $\nabla u_L$ (resp. $\nabla u_R$) est le gradient de la solution dans la cellule $C_L$ (resp. $C_R$), $n_L=-1$ (resp. $n_R=1$) est la normale sortante de $C_L$ (resp. $C_R$) et $\text{fields}_L$ (resp. $\text{fields}_R$) est le vecteur des champs dans la cellule $C_L$ (resp. $C_R$).

**Assemblage local (`assembly_local_flux_term`) :**

Pour la face $F_l$ et la $j$-ème fonction test, la contribution du flux est intégrée sur la face (intégration surfacique) :

$$\mathcal{F}_{l,j} \approx \sum_q w_q\, \texttt{flux}\!\left(\text{var}_L,\ \text{var}_R\right) \in \mathbb{R}^{n_u}$$

où `flux` désigne le callable de `AbstractFlux.__call__(varL, varR)`.

La fonction retourne les contributions aux deux mailles voisines avec **signe opposé** (le même `summed_flux`, calculé une seule fois par face) :

$$\text{résidu de } C_L \mathrel{-}= \mathcal{F}_{l,j} \qquad \text{résidu de } C_R \mathrel{+}= \mathcal{F}_{l,j}$$

Ce signe opposé vient de l'intégration par parties dans la formulation DG continue : le terme de bord apparaît en soustraction pour $C_L$ et en addition pour $C_R$.

### Termes de volume

**Formulation faible locale :**

Considérons $k\in\{0,\dots,n_c-1\}$ (`idx`) un indice de cellule.

Pour chaque maille $C_k$, on cherche les degrés de liberté $U_k = (u_{k,i})_i$ tels que :

$$\underbrace{\int_{C_k} B(u_h, \bar{\varphi}_{k,j})}_{b_{k,j}} = \underbrace{\int_{C_k} L(\bar{\varphi}_{k,j})}_{l_{k,j}} \quad \forall j \in \{0, \ldots, n_b - 1\}$$

où :
- $B(\cdot, \cdot)$ est la **forme bilinéaire** (`bilinear_form`) : dépend de la solution $u_h$ (trial) et de la fonction test $v$
- $L(\cdot)$ est la **forme linéaire** (`linear_form`) : dépend uniquement de la fonction test $v$

**Assemblage local (`assembly_local_volume_terms`) :**

La fonction calcule, pour la maille $C_k$ et la $j$-ème fonction test $\bar{\varphi}_{k,j}$, les deux quantités scalaires suivantes :

$$b_{k,j} = \int_{C_k} B(u_h, \bar{\varphi}_{k,j}) \approx \sum_q w_q\, B(u_h(x_q),\, \bar{\varphi}_{k,j}(x_q))$$

$$l_{k,j} = \int_{C_k} L(\bar{\varphi}_{k,j}) \approx \sum_q w_q\, L(\bar{\varphi}_{k,j}(x_q))$$

*Il n'y a pas d'indice trial $i$ séparé* : la solution $u_h|_{C_k}(x) = \sum_i u_{k,i}\, \bar{\varphi}_{k,i}(x)$ est déjà entièrement sommée dans `local_variables(idx)`. Ainsi, $b_{k,j}$ est directement le $j$-ème composant du produit matrice-vecteur $A_k U_k$ (pas la matrice $A_k$ elle-même).

### Assemblage du schéma complet

`assembly_scheme()` combine les termes de volume et de flux en un seul vecteur résidu de taille $n_c \cdot n_b \cdot n_u$.

**1. Termes de volume** (vmap sur les cellules et les fonctions test) :

```python
res = jax.vmap(
    lambda idx: jax.vmap(
        lambda j: scheme_pytree.assembly_local_full_volume_term(idx, j)
    )(basis_indices)           # vmap sur j ∈ [0, n_b-1]
)(jnp.arange(mesh.n_cells))   # vmap sur k ∈ [0, n_c-1]
# res : b_{k,j} - l_{k,j} pour chaque (k, j) -> (n_c, n_b, n_u)  
```

où `assembly_local_full_volume_term(idx, j)` retourne $b_{k,j} - l_{k,j}$ (résidu local).

**2. Termes de flux** (vmap sur les faces et les fonctions test) :

```python
(idxL, fluxL), (idxR, fluxR) = jax.vmap(
    lambda idf: jax.vmap(
        lambda j: scheme_pytree.assembly_local_flux_term(idf, j)
    )(basis_indices)            # vmap sur j ∈ [0, n_b-1]
)(jnp.arange(mesh.n_faces))    # vmap sur l ∈ [0, n_faces-1]
# fluxL, fluxR : (n_faces, n_b, n_u)
# idxL, idxR   : (n_faces, n_b)

res = res.at[idxL, basis_indices].add(fluxL)
res = res.at[idxR, basis_indices].add(fluxR)
```

**3. Reshape en vecteur :**

```python
res = jnp.reshape(res, (-1,))  # (n_c * n_b * n_u,)
```