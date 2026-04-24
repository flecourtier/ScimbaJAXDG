# Implémentation DG : Branche `scimba_jax_dg`

ANCIENNE VERSION !!!!!

## 6. Variables

> src/scimba_jax/linear_approximation/variables/variables_dg.py

En DG, on utilise deux écritures équivalentes de la variable discrète $u_h$ :

**Écriture locale (cellule par cellule)**

$$u_h|_{K_c}(x) = \sum_{i=0}^{n_b-1} u_{c,i}\, \bar{\varphi}_{c,i}(x), \qquad x \in K_c.$$

**Écriture globale (avec indicatrices)**

$$ u_h(x) = \sum_{c=0}^{n_c-1} \sum_{i=0}^{n_b-1} u_{c,i} \, \bar{\varphi}_{c,i}(x) \, \mathbb{1}_{\{x \in K_c\}}. $$

La première est la plus naturelle pour l'assemblage local ; la seconde met en évidence la définition globale sur $\Omega$ et la discontinuité entre cellules.

$$ u_h(x) = \sum_{k=0}^{n_c-1} \sum_{i=0}^{n_b-1} u_{k,i} \bar{\varphi}_{k,i}(x) \mathbb{1}_{\{x \in K_c\}} $$

### Degrés de liberté

Les degrés de liberté linéaires sont stockés dans `dofsl` de shape $(n_c, n_b, n_u)$.

| Attribut | Valeur |
|---|---|
| `ndof_linear` | $n_c \times n_b \times n_u$ (DOFs linéaires) |
| `ndof` | DOFs totaux incluant les paramètres des réseaux de neurones |

Les bases trial et test peuvent différer (Petrov-Galerkin) mais doivent avoir le même `out_dim` et `nb_basis`. Par défaut `test_basis = trial_basis`.

### Évaluation locale

Pour $x \in K_c$ :
$$ u_h(x) = \sum_{i=0}^{n_b-1} u_{k,i}\, \bar{\varphi}_{k,i}(x) = \texttt{einsum}(\text{"iv,qiv->qv"},\, \theta_k,\, b_k(x)) $$

où $\theta_k = \texttt{dofsl}[k]$ de shape $(n_b, n_u)$ et $b_k(x)$ de shape $(1, n_b, n_u)$.

### Méthodes d'évaluation

| Méthode | Input | Output | Note |
|---|---|---|---|
| `local_evaluate(point)` | $(d,)$ | $(n_u,)$ | 1 point, cherche la cellule |
| `evaluate(inputs)` | $(N, d)$ | $(N, n_u)$ | batch de points arbitraires |
| `evaluate_quad(x)` | $(n_c, n_{\text{quad}}, d)$ | $(n_c, n_{\text{quad}}, n_u)$ | points de quadrature du maillage |

Chaque méthode a une variante `_pure` (méthode statique) qui prend `variables_pytree` comme argument explicite — nécessaire pour la différentiation par rapport aux paramètres.

### Projection de $f$ sur l'espace DG

#### Sans post-processing (classique)

On cherche $U_k$ tel que $\forall j$ :

$$\underbrace{\int_{K_c} f \cdot \bar{\varphi}_{k,j}^{\text{test}}}_{b_{k,v,j}} = \sum_{i=0}^{n_b-1} u_{k,i} \underbrace{\int_{K_c} \bar{\varphi}_{k,i}^{\text{trial}} \cdot \bar{\varphi}_{k,j}^{\text{test}}}_{M_{k,v,ij}}$$

La matrice de masse est calculée par variable $v$ :
$$M_{k,v,ij} = \sum_q w_q\, \varphi^{\text{test}}_{k,q,i,v}\, \varphi^{\text{trial}}_{k,q,j,v}$$

Résolue cellule par cellule via `jnp.linalg.solve` (un système $(n_b \times n_b)$ par composante $v$), vmap sur les cellules.

#### Avec post-processing local (non-linéaire)

Si `post_processing.linear = True` : 1 itération Newton (résolution directe). Sinon : `n_iter = 20` itérations.

Le résidu local à minimiser pour la cellule $k$ :
$$R_{k,v,j}(\theta_k) = \int_{K_c} \left[\mathcal{P}(u_h|_{K_c}) - f\right] \bar{\varphi}_{k,j}^{\text{test}} \approx \sum_q w_q \left[\mathcal{P}(u_h(x_q)) - f(x_q)\right] \varphi^{\text{test}}_{k,q,j,v}$$

Newton : $\theta_k^{(s+1)} = \theta_k^{(s)} + \Delta\theta_k$ avec $\Delta\theta_k = -J_R^{-1} R_k(\theta_k^{(s)})$ via `jnp.linalg.lstsq`. Le VJP est défini via `jax.custom_vjp` (différentiation implicite).

### API publique

| Méthode | Description |
|---|---|
| `project(f)` | Projette $f$, met à jour `dofsl` |
| `project_jit(f)` | Idem avec JIT (compilé à la première appel, mis en cache) |


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