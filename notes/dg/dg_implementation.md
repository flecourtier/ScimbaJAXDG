# Implémentation DG : Branche `scimba_jax_dg`

<span style="color: blue;">Le fichier "implementation.md" contient toutes les informations communes à l'implémentation du schéma DG et du schéma FE.</span>

<span style="color: red;">TODO : Définir les espaces d'approximation.</span>

## Notations

<span style="color: blue;">On reprend toutes les notations utilisées dans le document principal.</span>

On définit également les notations suivantes pour l'implémentation spécifique à la méthode DG :

- $u_{c,i,\alpha}$ : DOF linéaire associé à la $i$-ème fonction de base de la $c$-ème cellule et à la composante $\alpha$-ème de la solution
- $\mathbf{u}_{c,\alpha} = (u_{c,i,\alpha})_{i=0}^{n_b-1}$ : vecteur des DOFs linéaires associés à la composante $\alpha$-ème de la solution dans la cellule $c$

## Variables

> src/scimba_jax/linear_approximation/variables/variables_dg.py

On considère donc un système de $n_\text{out}$ équations couplées à résoudre sur un domaine $\Omega \subset \mathbb{R}^d$ (ex : système de Navier-Stokes avec $n_\text{out}=4$ en 2D : vélocité (de taille $d$), pression, température).
On définit la solution discrète $U_h : \Omega \to \mathbb{R}^{n_\text{out}}$ (reconstruite à partir des DOFs linéaires) et $U_{h,\alpha} : \Omega \to \mathbb{R}$ sa $\alpha$-ème composante, $\alpha \in \{0, \ldots, n_\text{out}-1\}$.

En DG, on utilise deux écritures équivalentes de cette variable discrète :

- **Écriture locale (cellule par cellule) :**

    $$U_{h,\alpha}|_{g(K_c)}(X) = \sum_{i=0}^{n_b-1} u_{c,i,\alpha}\, \bar{\varphi}_{c,i,\alpha}(X), \qquad X \in g(K_c).$$

- **Écriture globale (avec indicatrices) :**

    $$U_{h,\alpha}(X) = \sum_{c=0}^{n_\text{cells}-1} \sum_{i=0}^{n_b-1} u_{c,i,\alpha} \, \bar{\varphi}_{c,i,\alpha}(X) \, \mathbb{1}_{\{X \in g(K_c)\}}, \qquad X \in \Omega.$$

avec $u_{c,i,\alpha}$ les DOFs linéaires associés à la $i$-ème fonction de base de la $c$-ème cellule et à la composante $\alpha$-ème de la solution, et $\bar{\varphi}_{c,i,\alpha}$ la $i$-ème fonction de base (trial) associée à la composante $\alpha$-ème de la solution dans la cellule $c$.

### Degrés de liberté

Les degrés de liberté linéaires sont stockés dans `dofsl` de shape $(n_\text{cells}, n_b, n_\text{out})$.

| Attribut | Valeur |
|---|---|
| `ndof_linear` | nombre de DOFS linéaires ($n_\text{cells} \times n_b \times n_\text{out}$) |
| `ndof` | nombre de DOFs totaux (incluant les paramètres des réseaux de neurones) |

### Méthodes d'évaluation

| Méthode | Input | Output | Note |
|---|---|---|---|
| `local_evaluate(inputs)` | $(d,)$ | $(n_\text{out},)$ | 1 point (cherche la cellule) |
| `evaluate(inputs)` | $(\cdot, d)$ | $(\cdot, n_\text{out})$ | batch de points arbitraires |
| `evaluate_quad(inputs)` | $(n_\text{cells}, n_{\text{quad}}, d)$ | $(n_\text{cells}, n_{\text{quad}}, n_\text{out})$ | points de quadrature du maillage |

Chaque méthode a une variante `_pure` (méthode statique) qui prend `variables_pytree` comme argument explicite — nécessaire pour la différentiation par rapport aux paramètres.

### Projection d'une fonction sur l'espace DG

On suppose ici qu'on veut projeter une fonction $f : \Omega \to \mathbb{R}^{n_\text{out}}$ sur l'espace DG.

<!-- Pour simplifier, on fixe $n_\text{out} = 1$ (une seule composante) dans les formules suivantes. -->

#### Sans post-processing (classique)

Pour une cellule physique $g(K_c)$ et une composante $\alpha$-ème de la solution, on cherche $\mathbf{u}_{c,\alpha} = (u_{c,i,\alpha})_{i=0}^{n_b-1}$ tel que :

$$\underbrace{\int_{g(K_c)} f(X) \cdot \bar{\psi}_{c,j,\alpha}(X)dX}_{(b_{c,\alpha})_j} = \sum_{i=0}^{n_b-1} u_{c,i,\alpha} \underbrace{\int_{g(K_c)} \bar{\varphi}_{c,i,\alpha}(X) \cdot \bar{\psi}_{c,j,\alpha}(X)dX}_{(M_{c,\alpha})_{i,j}}, \quad \forall j,$$

La matrice de masse $M_{c,\alpha}$ ainsi que le vecteur de projection $b_{c,\alpha}$ sont calculés via des intégrations numériques (quadrature) sur la cellule physique $g(K_c)$, comme décrit dans le document principal.
Le système $M_{c,\alpha}\, \mathbf{u}_{c,\alpha} = \mathbf{b}_{c,\alpha}$ est alors résolu cellule par cellule via `jnp.linalg.solve` (un système $(n_b \times n_b)$ par composante $\alpha$), vmap sur les cellules.

#### Avec post-processing local (possiblement non-linéaire)

> **Remarques :**
> - Le post-processing local est appliqué à la variable discrète reconstruite $U_h$ (et non pas directement aux DOFs linéaires) : $\bar{U_h}(X) = \mathcal{P}(U_h(X), X)$.
> - Le post-processing peut coupler les différentes composantes de la solution, ce qui implique qu'il est nécessaire de résoudre un système global sur toutes les composantes $\alpha$-ème à la fois (pas de découplage composante par composante possible comme dans le cas classique sans post-processing).
> - Comme le post-processing peut être non-linéaire, on doit faire un Newton. Si le post-processing est linéaire, on peut faire une seule itération du Newton.

Pour une cellule physique $g(K_c)$ fixée, on cherche $\mathbf{u}_c = (u_{c,i,\alpha})_{i,\alpha}$ tel que :

$$R_{j,\alpha}(\mathbf{u}_c) := \int_{g(K_c)} \left[\mathcal{P}(U_h(X), X) - f(X)\right]_\alpha \cdot \bar{\psi}_{c,j,\alpha}(X)\, dX = 0, \quad \forall j \in \{0,\ldots,n_b-1\},\, \alpha \in \{0,\ldots,n_\text{out}-1\}$$

avec $U_h(X)$ la solution discrète reconstruite à partir des DOFs linéaires $\mathbf{u}_c$.

En notant $\mathbf{r}(\mathbf{u}_c) = (R_{j,\alpha}(\mathbf{u}_c))_{j,\alpha} \in \mathbb{R}^{n_b \times n_\text{out}}$, ce système non-linéaire est résolu par Newton :

$$\mathbf{u}_c^{(s+1)} = \mathbf{u}_c^{(s)} - J_\mathbf{r}^{-1}\, \mathbf{r}(\mathbf{u}_c^{(s)})$$

via `jnp.linalg.lstsq`. Le VJP est défini via `jax.custom_vjp` (différentiation implicite).

> **Exemple (couplage des composantes) :** Prenons $n_\text{out} = 2$ et un post-processing simple (linéaire) :
> 
> $$\bar{U}_h(X) = \mathcal{P}(U_h(X),X) = (U_{h,0}(X) + U_{h,1}(X) , U_{h,1}(X))$$
> Le résidu sur la composante $\alpha=0$ est :
> $$R_{j,0}(\mathbf{u}_c) = \int_{g(K_c)} \left[U_{h,0}(X) + U_{h,1}(X) - f_0(X)\right] \cdot \bar{\psi}_{c,j,0}(X)\, dX$$
> $$= \int_{g(K_c)} \left[\sum_i u_{c,i,0}\, \bar{\varphi}_{c,i,0}(X) + \sum_i u_{c,i,1}\, \bar{\varphi}_{c,i,1}(X) - f_0(X)\right] \cdot \bar{\psi}_{c,j,0}(X)\, dX$$
> Et ainsi $\frac{\partial R_{j,0}}{\partial u_{c,i,1}} \neq 0$, ce qui signifie que le résidu sur la composante $0$ dépend des DOFs de la composante $1$ via le post-processing $\mathcal{P}$. C'est pourquoi, il est nécessaire de résoudre un système global sur toutes les composantes à la fois.
>
> En pratique, si on est dans le cas $\mathcal{P}_\theta$ (un réseau de neurones), toutes les composantes sont mélangées dans les couches cachées, donc tous les blocs hors-diagonale du Jacobien sont non nuls a priori. 

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

Considérons $n_f=n_\text{cells}+1$ le nombre de faces et $l \in \{1, ..., n_f-2\}$ (`idf`) un indice de face intérieure.

On définit $C_L$ et $C_R$ comme les deux cellules voisines (gauche et droite) de la face $F_l$ .

Le flux numérique est défini par la classe abstraite `AbstractFlux` via `__call__(varL, varR)` où `varL` (resp. `varR`) est définit par :

$$\text{var}_L = (u_L,\ v_L,\ \nabla u_L,\ \nabla v_L,\ n_L,\ \text{fields}_L)$$

où $u_L$ (resp. $u_R$) est la solution dans la cellule $C_L$ (resp. $C_R$), $\nabla u_L$ (resp. $\nabla u_R$) est le gradient de la solution dans la cellule $C_L$ (resp. $C_R$), $n_L=-1$ (resp. $n_R=1$) est la normale sortante de $C_L$ (resp. $C_R$) et $\text{fields}_L$ (resp. $\text{fields}_R$) est le vecteur des champs dans la cellule $C_L$ (resp. $C_R$).

**Assemblage local (`assembly_local_flux_term`) :**

Pour la face $F_l$ et la $j$-ème fonction test, la contribution du flux est intégrée sur la face (intégration surfacique) :

$$\mathcal{F}_{l,j} \approx \sum_q w_q\, \texttt{flux}\!\left(\text{var}_L,\ \text{var}_R\right) \in \mathbb{R}^{n_\text{out}}$$

où `flux` désigne le callable de `AbstractFlux.__call__(varL, varR)`.

La fonction retourne les contributions aux deux mailles voisines avec **signe opposé** (le même `summed_flux`, calculé une seule fois par face) :

$$\text{résidu de } C_L \mathrel{-}= \mathcal{F}_{l,j} \qquad \text{résidu de } C_R \mathrel{+}= \mathcal{F}_{l,j}$$

Ce signe opposé vient de l'intégration par parties dans la formulation DG continue : le terme de bord apparaît en soustraction pour $C_L$ et en addition pour $C_R$.

### Termes de volume

**Formulation faible locale :**

Considérons $k\in\{0,\dots,n_\text{cells}-1\}$ (`idx`) un indice de cellule.

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

`assembly_scheme()` combine les termes de volume et de flux en un seul vecteur résidu de taille $n_\text{cells} \cdot n_b \cdot n_\text{out}$.

**1. Termes de volume** (vmap sur les cellules et les fonctions test) :

```python
res = jax.vmap(
    lambda idx: jax.vmap(
        lambda j: scheme_pytree.assembly_local_full_volume_term(idx, j)
    )(basis_indices)           # vmap sur j ∈ [0, n_b-1]
)(jnp.arange(mesh.n_\text{cells}ells))   # vmap sur k ∈ [0, n_\text{cells}-1]
# res : b_{k,j} - l_{k,j} pour chaque (k, j) -> (n_\text{cells}, n_b, n_\text{out})  
```

où `assembly_local_full_volume_term(idx, j)` retourne $b_{k,j} - l_{k,j}$ (résidu local).

**2. Termes de flux** (vmap sur les faces et les fonctions test) :

```python
(idxL, fluxL), (idxR, fluxR) = jax.vmap(
    lambda idf: jax.vmap(
        lambda j: scheme_pytree.assembly_local_flux_term(idf, j)
    )(basis_indices)            # vmap sur j ∈ [0, n_b-1]
)(jnp.arange(mesh.n_faces))    # vmap sur l ∈ [0, n_faces-1]
# fluxL, fluxR : (n_faces, n_b, n_\text{out})
# idxL, idxR   : (n_faces, n_b)

res = res.at[idxL, basis_indices].add(fluxL)
res = res.at[idxR, basis_indices].add(fluxR)
```

**3. Reshape en vecteur :**

```python
res = jnp.reshape(res, (-1,))  # (n_\text{cells} * n_b * n_\text{out},)
```