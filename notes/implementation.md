# Implémentation : Branche `scimba_jax_dg`

---
Dans ce fichier, on présente l'implémentation commune aux deux méthodes : FEM et DG. Deux fichiers supplémentaires (`fem_implementation.md` et `dg_implementation.md`) détaillent les spécificités de chacune.

On se place dans le contexte où on travaille **sur un seul "patch"** divisé en plusieurs cellules. L'élargissement à des domaines composés de plusieurs patchs sera abordée par la suite.


## Notations

### Générales

- $\Omega$ : domaine physique
- $d$ : dimension spatiale
- $\ell$ : indice de direction spatiale ($\ell \in \{0, \ldots, d-1\}$)
- $\theta$ : paramètres du/des réseau.x de neurones (notation générique pour le mapping, le post-processing et les bases apprenables)

### Mapping

- $g, g_\theta$ : mapping du maillage (composition de $m$ fonctions), analytique ou réseau de neurones
- $m$ : nombre de fonctions composées dans le mapping

### Quadrature

- $q$ : ordre de quadrature par direction
- $n_\text{quad} = q^d$ : nombre total de points de quadrature

### Maillage

- $n_\text{cells} = \prod_{\ell=0}^{d-1} N_\ell$ : nombre total de cellules (`n_cells_total`)
- $N_\ell$ : nombre de cellules dans la direction $\ell$
- $N_\text{cells} = (N_0, \ldots, N_{d-1})$ : tuple du nombre de cellules par direction (`n_cells`)
- $n_\text{faces}$ : nombre total de faces (`n_faces`)
- $n_{f,\ell}$ : nombre de faces du groupe $\ell$ (normales à la direction $\ell$)
- $\mathcal{T}_h$ : maillage uniforme du domaine logique $[0,1]^d$
- $K_\text{ref} = [0,1]^d$ : élément de référence
- $K_c$ : la $c$-ème cellule du maillage, 
    - $c \in [0, n_\text{cells} - 1]$ : indexation plate
    - $\bar{c} = (c_0, \ldots, c_{d-1})$ : indexation multi-dim (avec $c_\ell \in [0, N_\ell - 1]$)
- $\hat{\xi}$, $\xi$, $X$ : points dans les espaces de référence, logique et physique respectivement
- $\hat{\xi}_i$, $w_i$ : $i$-ème point et poids de quadrature volumique sur l'élément de référence
- $\hat{\xi}_i^s$, $w_i^s$ : $i$-ème point et poids de quadrature surfacique sur la face de référence
- $\bar{w}_i$, $\bar{w}_i^s$ : poids corrigés pour la cellule $K_c$ et la face $F_k$ respectivement 
- $T_{K_c}$ : transformation géométrique de l'élément de référence vers une cellule $K_c$
- $\Phi = g \circ T_{K_c}$ : composition du mapping et de la transformation géométrique, soit le passage de l'élément de référence au domaine physique
- $F_\text{ref} = [0,1]^{d-1}$ : face de référence (hyperplan dans $[0,1]^d$)
- $F_k$ : la $k$-ème face du maillage
    - $k \in [0, n_\text{faces} - 1]$ : indexation plate
    - $\bar{k} = (k_0, \ldots, k_{d-1})$ : indexation multi-dim (avec $k_\ell \in [0, N_\ell]$) pour un groupe de faces $\ell$ fixé

### Bases

- $n_b$ : nombre de fonctions de base par cellule (`nb_basis`)
- $n_{\text{out}}$ : dimension de sortie, nombre de variables dans le système (`out_dim`) 
    <span style="color: red;">-> à clarifier</span>
- $\bar{\varphi}_{c,i}, \bar{\psi}_{c,i}$ : notations génériques pour les fonctions de base (trial) et test de la $c$-ème cellule, respectivement.
- $\varphi_{c,i}$, $\varphi_{c,i}^{\theta,P}$, $\varphi_{c,i}^{\theta,C}$ : la $i$-ème fonction de base (trial) dans la $c$-ème cellule, respectivement : analytique, un réseau de neurones (`Patchwise`), $n_\text{cells}$ réseaux de neurones (`Cellwise`)

### Variables discrètes

- $u_h$ : variable discrète reconstruite à partir des DOFs linéaires et des fonctions de base

### Post-processing

- $\mathcal{P}, \mathcal{P}_\theta$ : post-processing, resêctivement : analytique, réseau de neurones
- $\bar{u_h}$ : variable discrète post-processée

## 1. Mapping

> src/scimba_jax/mapping/mapping.py

### Description

Le mapping $g$ est utilisé pour transformer les points de quadrature du domaine logique $[0,1]^d$ vers le domaine physique $\Omega$ via $X = g(\xi)$ avec $\xi$ un point logique et $X$ son image dans le domaine physique. 

Un `Mapping` est une composition ordonnée de mapping :

$$g = g_m \circ g_{m-1} \circ \cdots \circ g_1$$

avec $m$ le nombre de fonctions composées où chaque $g_j$, $j \in \{1, \ldots, m\}$, est un mapping inversible d'un des deux types suivants :

| Type | Description |
|---|---|
| `InvertibleFunction` | Mapping inversible analytique |
| `InvertibleNet` | Réseau inversible du module Equinox (`eqx.Module`) |

L'application inverse de $g$ est donc donnée par :

$$g^{-1} = g_1^{-1} \circ g_2^{-1} \circ \cdots \circ g_m^{-1}.$$

### Calcul du Jacobien

<span style="color: red;">A COMPLETER</span>

### Méthodes disponibles

Les méthodes disponibles pour le `Mapping` sont : mapping direct, inverse, jacobien, jacobien inverse, déterminant du jacobien, déterminant du jacobien inverse.

### Enregistrement comme pytree JAX

`Mapping` est enregistré comme pytree jax avec les attributs suivants :
- `children` : les `InvertibleNet` (apprenables, mis à jour par l'optimiseur)
- `aux_data` : les `InvertibleFunction` statiques (non différentiables au sens pytree)

## 2. Quadrature

> src/scimba_jax/linear_approximation/quad/gauss_quad.py

### Description

La classe `UnitSquareTensorized` construit une quadrature par produit tensoriel sur l'hypercube de référence $K_\text{ref} = [0,1]^d$.

$$ \int_{K_\text{ref}} f(\hat{\xi})\, d\hat{\xi} \approx \sum_{i=1}^{n_{\text{quad}}} w_i\, f(\hat{\xi}_i) $$

où $n_{\text{quad}} = q^d$ est le nombre total de points volumiques, $w_i$ sont les poids de quadrature et $\hat{\xi}_i$ les points de quadrature.

### Construction des points et poids

Actuellement, deux variantes de règles de quadrature 1D (remappés de l'intervalle $[-1,1]$ à $[0,1]$) sont disponibles : Gauss-Legendre et Chebyshev. Pour un ordre de quadrature $q$, les polynômes de Gauss-Legendre permettent d'intégrer exactement les polynômes de degré jusqu'à $2q-1$.

Ces points et poids 1D sont ensuite tensorisés pour construire les points et poids volumiques et surfaciques sur l'hypercube de référence. Plus précisément, les points volumiques sont construits par le produit tensoriel des points 1D dans chaque direction, et les poids volumiques sont le produit des poids 1D correspondants. Pour les points surfaciques, on fixe une coordonnée normale à $0$ ou $1$ et on construit un produit tensoriel $(d-1)$-dimensionnel dans les directions tangentes.

### Dimensions des tenseurs

* `points_1d`, `weights_1d` : $(q,)$, $(q,)$
* `volumic_points`, `volumic_weights` : $(q^d,\, d)$, $(q^d,)$
* `surfacic_points`, `surfacic_weights` : $(2d,\, q^{d-1},\, d)$, $(2d,\, q^{d-1})$

## 3. Maillage

> src/scimba_jax/linear_approximation/meshes/mesh.py

### Structure

On définit un maillage uniforme $\mathcal{T}_h$ sur $[0,1]^d$, découpé en $N_\ell$ cellules dans la direction $\ell \in \{0, \ldots, d-1\}$.

On définit $N_\text{cells} = (N_0, \ldots, N_{d-1})$ comme le tuple du nombre de cellules par direction et $n_\text{cells} = \prod_{\ell=0}^{d-1} N_\ell$ le nombre total de cellules. 

Chaque cellule $K_c$ (où $c$ est l'indice de la cellule), définie comme un hypercube de taille $1/N_\ell$ dans la direction $\ell$ peut être identifiée par deux représentations équivalentes :

| Indexation | Notation | Valeurs | Conversion |
|---|---|---|---|
| Plate (`fidx`) | $c \in [0, n_\text{cells} - 1]$ | entier | `_midx_to_fidx` |
| Multi-dim (`midx`) | $\bar{c} = (c_0, \ldots, c_{d-1})$ avec $c_\ell \in [0, N_\ell - 1]$ | tuple | `_fidx_to_midx` |


**Faces :** Les faces sont des hypersurfaces $(d-1)$-dimensionnelles, organisées en $d$ groupes selon leur axe normal. Les faces du groupe $\ell$ sont perpendiculaires à cette même direction ce qui signifie que chaque couche (correspondant à une position fixée sur l'axe $\ell$) contient $\prod_{j \neq \ell} N_j$ faces, et il y a $N_\ell + 1$ couches, soit au total $n_{f,\ell}$ faces dans le groupe $\ell$ :

$$n_{f,\ell} = (N_\ell + 1) \cdot \prod_{j \neq \ell} N_j$$

Ainsi le nombre total de faces est la somme sur les groupes :

$$n_\text{faces} = \sum_{\ell=0}^{d-1} n_{f,\ell} = \sum_{\ell=0}^{d-1} (N_\ell + 1) \cdot \prod_{j \neq \ell} N_j$$

> **Exemple :** pour un maillage 3D avec $N_\text{cells} = (2, 2, 3)$ cellules.
> - $\ell = 0$ (axe $x$) : $n_{f,0} = (2+1) \cdot (2 \cdot 3) = 18$ faces (bleu)
> - $\ell = 1$ (axe $y$) : $n_{f,1} = (2+1) \cdot (2 \cdot 3) = 18$ faces (rouge)
> - $\ell = 2$ (axe $z$) : $n_{f,2} = (3+1) \cdot (2 \cdot 2) = 16$ faces (vert)
> - total : $n_\text{faces} = 18 + 18 + 16 = 52$ faces
> 
> <img src="images/faces_x.png" width="300"> <img src="images/faces_y.png" width="300"> <img src="images/faces_z.png" width="300">

De manière équivalente aux cellules, chaque face $F_k$ (où $k$ est l'indice de la face) peut être identifiée par deux représentations équivalentes :

| Indexation | Notation | Valeurs | Conversion |
|---|---|---|---|
| Plate (`face_fidx`) | $k \in [0, n_\text{faces} - 1]$ | entier | / |
| Multi-dim (`face_midx`) + groupe $\ell$ | $\bar{k} = (k_0, \ldots, k_{d-1})$ avec $k_\ell \in [0, N_\ell]$ | tuple | `_face_fidx_to_face_midx_and_face_type` |

> **Exemple :** pour le même maillage 3D avec $N_\text{cells} = (2, 2, 3)$ cellules.
> - $k=0$ (première face du groupe $\ell=0$) $\rightarrow \bar{k} = (0, 0, 0)$
> - $k=17$ (dernière face du groupe $\ell=0$) $\rightarrow \bar{k} = (2, 1, 2)$
> - $k=18$ (première face du groupe $\ell=1$) $\rightarrow \bar{k} = (0, 0, 0)$

Les faces sont ensuite classées en deux catégories :
- **Internes** (`internal_faces_idx`) : couches intermédiaires ($1 \leq$ position $\leq N_\ell - 1$), partagées par deux cellules. 
    
    Si $N_\ell = 1$ dans une direction, il n'y a aucune face interne dans le groupe $\ell$.
- **Externes** (`external_faces_idx`) : première et dernière couche (position $0$ et $N_\ell$), faces de bord.

La fonction `_face_fidx_to_neighbors_fidx` retourne les indices plats des deux cellules voisines d'une face. Pour les faces de bord, le voisin extérieur est encodé par une valeur négative.

### Chaîne de mapping

On note $T_{K_c} : K_\text{ref} \to K_c$ la transformation affine qui envoie l'élément de référence sur la cellule $K_c$ :

$$\xi = T_{K_c}(\hat{\xi}) = \frac{\hat{\xi} + \bar{c}}{N_\text{cells}}, \qquad \hat{\xi} = T_{K_c}^{-1}(\xi) = \xi \cdot N_\text{cells} - \bar{c}$$

où la division et la multiplication se font composante par composante.

> **Remarque :** En développant composante par composante, on a pour la $\ell$-ème composante :
> $$(T_{K_c}(\hat{\xi}))_\ell = \frac{\hat{\xi}_\ell + c_\ell}{N_\ell}$$
> Son jacobien est donc une matrice diagonale :
> $$J_{T_{K_c}} = \text{diag}\left(\frac{1}{N_0}, \frac{1}{N_1}, \ldots, \frac{1}{N_{d-1}}\right)$$
> Et son déterminant :
> $$|\det J_{T_{K_c}}| = \prod_{\ell=0}^{d-1} \frac{1}{N_\ell} = \frac{1}{n_\text{cells}}$$

Les points de quadrature traversent alors trois espaces successifs :

$$\hat{\xi} \in K_\text{ref} = [0,1]^d \xrightarrow{\ T_{K_c}\ } \xi \in K_c \subset [0,1]^d \xrightarrow{\ g\ } X \in \Omega$$

Autrement dit, les points de quadrature sont d'abord transformés de l'élément de référence à la cellule correspondante du maillage logique (via une transformation affine), puis du maillage logique au domaine physique via le mapping $g$.

La première flèche est réalisée par `_unit_hypercube_to_cell`, la seconde par `mapping.local_mapping`.

### Normales physiques (formule de Nanson)

La formulation DG requiert la normale sortante unitaire sur chaque face dans le domaine physique $\Omega$. Dans le domaine logique, la normale d'une face du groupe $\ell$ est simplement le vecteur canonique $e_\ell = (0, \ldots, 0, 1, 0, \ldots, 0)$ avec un $1$ à la position $\ell$. On obtient la normale physique via la formule de Nanson :

$$n_{\text{phys}} = \frac{J_g(\xi)^{-T}\, e_\ell}{\|J_g(\xi)^{-T}\, e_\ell\|}$$

avec $\|\cdot\|$ la norme euclidienne.

### Intégrations volumiques ($d$-dimensions)

On note $\Phi = g \circ T_{K_c} : K_\text{ref} \to \Omega$ la composition des deux transformations, soit le passage de l'élément de référence au domaine physique. 

Son jacobien est donné par la règle de la chaîne :

$$|\det J_{\Phi}(\hat{\xi})| = |\det J_g(T_{K_c}(\hat{\xi}))| \cdot |\det J_{T_{K_c}}| = \frac{1}{n_\text{cells}}\,|\det J_g(T_{K_c}(\hat{\xi}))|$$

Le changement de variable sur la cellule physique $g(K_c)$ se fait en deux étapes :

$$\int_{g(K_c)} f(X)\, dX \underbrace{=}_{X = g(\xi)} \int_{K_c} f(g(\xi))\, |\det J_g(\xi)|\, d\xi \underbrace{=}_{\xi = T_{K_c}(\hat{\xi})} \frac{1}{n_\text{cells}} \int_{K_\text{ref}} f(\Phi(\hat{\xi}))\, |\det J_g(T_{K_c}(\hat{\xi}))|\, d\hat{\xi}$$

Discrétisé par quadrature sur $K_\text{ref}$ (points $\hat{\xi}_i$, poids $w_i$) :

$$ \int_{g(K_c)} f(X)\, dX \approx \sum_{i=1}^{n_{\text{quad}}} \bar{w_i}\, f(\Phi(\hat{\xi}_i)) $$

où $\bar{w_i} = \frac{1}{n_\text{cells}}\, w_i\, |\det J_g(T_{K_c}(\hat{\xi}_i))|$ sont les poids corrigés pour la cellule $K_c$.

### Intégrations surfaciques ($(d-1)$-dimensions)

Le changement de variable sur la face physique $g(F_k)$, avec $F_k$ une face du groupe $\ell$, se fait en deux étapes. Contrairement au cas volumique, l'élément de surface physique fait intervenir le facteur de Nanson $\|J_g^{-T} e_\ell\|$ en plus du jacobien :

$$\int_{g(F_k)} f\, dS \underbrace{=}_{X = g(\xi^s)} \int_{F_k} f(g(\xi^s))\, |\det J_g(\xi^s)|\,\|J_g(\xi^s)^{-T} e_\ell\|\, d\xi^s \underbrace{=}_{\xi = T_{K_c}(\hat{\xi}^s)} \frac{N_\ell}{n_\text{cells}} \int_{F_\text{ref}} f(\Phi(\hat{\xi}^s))\, |\det J_g(T_{K_c}(\hat{\xi}^s))|\, \|J_g(T_{K_c}(\hat{\xi}^s))^{-T} e_\ell\|\, d\hat{\xi}^s$$

avec $\|\cdot\|$ la norme euclidienne et $F_\text{ref} = [0,1]^{d-1}$ la face de référence (hyperplan dans $[0,1]^d$).

Discrétisé par quadrature sur $F_\text{ref}$ (points $\hat{\xi}_i^s$, poids $w_i^s$) :

$$\int_{g(F_k)} f\, dS \approx \sum_i \bar{w}_i^s\, f(\Phi(\hat{\xi}_i^s))$$

où $\bar{w}_i^s = \dfrac{N_\ell}{n_\text{cells}}\, w_i^s\, |\det J_g(T_{K_c}(\hat{\xi}_i^s))|\, \|J_g(T_{K_c}(\hat{\xi}_i^s))^{-T} e_\ell\|$ sont les poids corrigés pour la face $F_k$.

> **Exemple :** Prenons $g(\xi) = (a\xi_0, b\xi_1)$ avec $a \neq b$, $J_g = \text{diag}(a, b)$, $|\det J_g| = ab$.
>
> Considérons une face verticale (appartenant au groupe $\ell=0$) du domaine logique (de longueur $1$), par exemple la face $\xi_0 = 1$. Après mapping, ce segment a comme longueur $b$ (ce qui signifie que seul $\xi_1$ s'étire).
>
> Le scaling correct de cette face est donc $b$, mais $|\det J_g| = ab \neq b$ : le jacobien volumique surestime d'un facteur $a$ (l'étirement dans la direction normale $e_0$, qui ne contribue pas à l'aire de la face).
>
> Le facteur de Nanson corrige exactement cela : $J_g^{-T} e_0 = e_0/a$, donc $\|J_g^{-T} e_0\| = 1/a$, et :
> $$|\det J_g| \cdot \|J_g^{-T} e_0\| = ab \cdot \frac{1}{a} = b \checkmark$$

### Méthodes publiques

| Méthode | Retourne | Shape |
|---|---|---|
| `evaluate_mesh_points()` | points physiques sur toutes les cellules | $(n_\text{cells}, n_{\text{quad}}, d)$ |
| `evaluate_mesh_weights_points()` | poids corrigés + points physiques | $(n_\text{cells}, n_{\text{quad}})$, $(n_\text{cells}, n_{\text{quad}}, d)$ |
| `find_cell_index(x)` | indice `midx` et `fidx` de la cellule contenant $x$ | |
| `cell_centroid(c)` | centroïde physique de la cellule $c$ | $(d,)$ |

### Enregistrement comme pytree JAX

`Mesh` est enregistré comme pytree JAX avec :
- `children` : le `Mapping` (apprenable, mis à jour par l'optimiseur)
- `aux_data` : tous les autres attributs statiques (`dim`, `n_cells`, `ref_quad`, indices de faces, etc.)

## 4. Bases

> src/scimba_jax/linear_approximation/basis/general_bases.py

Commençons par introduire le nombre de variables dans le système PDE, noté $n_\text{out}$ (par exemple $n_\text{out}=1$ pour une équation scalaire, $n_\text{out}=d$ pour un système de Navier-Stokes incompressible).

On définit $n_b$ comme le nombre de fonctions de base par cellule (identique pour toutes les cellules). Pour un élément local $K_c$, on note la $i$-ème fonction de base (trial) de la $c$-ème cellule par $\bar{\varphi}_{c,i} : K_c \to \mathbb{R}^{n_{\text{out}}}$.

### Types de bases

Une fonction de base $\bar{\varphi}_{c,i}$ peut-être définie de trois manières différentes selon le `basis_type` choisi :

| `basis_type` | Définition mathématique | Contrainte | Description |
|---|---|---|---|
| `"scalar"` | $K_c \to \mathbb{R}$ | `out_dim == 1` | base scalaire |
| `"field"` | $K_c \to \mathbb{R}^d$ | `out_dim == dim` | base de champ vectoriel |
| `"vec"` | $K_c \to \mathbb{R}^{n_{\text{out}}}$ | `out_dim > 1` | base vectorielle quelconque |

### Construction d'une base

On considère qu'une base $\bar{\varphi}_{c,i}$ peut-être construites de trois manières différentes :

- **Analytique (notée $\varphi_{c,i}$) :** définie par une formule analytique (ex. Taylor, Lagrange) 
- **Patchwise (notée $\varphi_{c,i}^{\theta,P}$) :** définie par une base analytique multipliée par un même réseau de neurones $v_\theta$ partagé par toutes les cellules
  $$ \varphi_{c,i}^{\theta,P}(x) = v_\theta(x)\, \varphi_{c,i}(x) $$
- **Cellwise (notée $\varphi_{c,i}^{\theta,C}$) :** définie par une base analytique multipliée par un réseau de neurones $v_{c,\theta}$ spécifique à chaque cellule
  $$ \varphi_{c,i}^{\theta,C}(x) = v_{c,\theta}(x)\, \varphi_{c,i}(x) $$

Ces trois bases sont implémentées dans des classes distinctes (`AnalyticBasis`, `PatchwiseParametricBasis`, `CellwiseParametricBasis`) qui héritent de l'interface de `GeneralBasis`, en particulier :

| Méthode | Signature | Retourne |
|---|---|---|
| `__call__(c, inputs)` | `c` : indice de cellule, `inputs` : $(n_{\text{quad}}, d)$ | $(n_{\text{quad}}, n_b, n_{\text{out}})$ |
| `derivative(c, inputs)` | idem | $(n_{\text{quad}}, n_b, n_{\text{out}}, d)$ via `jax.jacobian` |

### Base analytique

> src/scimba_jax/linear_approximation/basis/analytic_bases.py

<span style="color: blue;">Pour le moment, deux types de bases analytiques sont implémentés : Taylor (DG) et Lagrange (FEM). On detaillera leur construction dans les fichiers dédiés à chacune des méthodes. 
</span>

### Fonctions tests

De manière équivalente, on peut introduire les fonctions tests $\bar{\psi}_{c,i} : K_c \to \mathbb{R}^{n_{\text{out}}}$, qui peuvent être différentes des fonctions de base (Petrov-Galerkin) mais doivent avoir le même `out_dim` et `nb_basis`. 

On considère que les fonctions tests sont construites de la même manière que les fonctions de base, c'est à dire qu'elles peuvent être analytiques, patchwise ou cellwise. Voir la table ci-dessous pour les différentes combinaisons possibles de bases trial et test :

| Trial | Test |
|---------|--------|
|`AnalyticBasis`| `AnalyticBasis`|
|`PatchwiseParametricBasis`| `AnalyticBasis`|
|`PatchwiseParametricBasis`| `PatchwiseParametricBasis`|
|`CellwiseParametricBasis`| `AnalyticBasis`|
|`CellwiseParametricBasis`| `CellwiseParametricBasis`|

### Enregistrement comme pytree JAX

Les bases sont enregistrées comme pytrees JAX (classes `AnalyticBasis`, `PatchwiseParametricBasis`, `CellwiseParametricBasis`) avec une séparation entre :
- `children` : les objets potentiellement apprenables et/ou différentiables (en particulier `mesh`, puis les modules `eqx.Module` quand présents)
- `aux_data` : les métadonnées statiques (`nb_basis`, `out_dim`, `basis_type`, `local_basis`, flags et callables statiques)

Pour `PatchwiseParametricBasis` et `CellwiseParametricBasis`, le module paramétrique n'est placé dans `children` que lorsqu'il est effectivement présent comme module Equinox ; sinon il est stocké dans `aux_data`.

## 5. Variables discrètes

<span style="color: blue;">
Pour les deux méthodes DG et EF, la variable discrète est définie de manière similaire à partir des fonctions de base et des DOFs linéaires. La différence entre les deux méthodes réside dans la structure globale de l'espace discret et les contraintes de continuité. Ces points seront détaillés dans les fichiers dédiés à chacune des méthodes.
</span>

Dans la suite du document, on notera $u_h$ la variable discrète définie de manière globale sur $\Omega$ (avec indicatrices).

## 6. Post-processing

> src/scimba_jax/linear_approximation/variables/postprocessing.py

Le post-processing, notéée $\mathcal{P}$, est un opérateur qui agit sur la variable discrète reconstruite à partir des DOFs. Cet opérateur peut-être une fonction analytique (linéaire ou non-linéaire) ou un réseau de neurones, un `eqx.Module` (non-linéaire).

On peut considérer deux types de processings :
- **Local** : la valeur post-processée en un point $x$ dépend uniquement de la valeur reconstruite $u_h|_{K_c}(x)$ (dans la cellule $K_c$ contenant $x$), c'est-à-dire que la solution post-processée s'écrit :
$$\bar{u_h}(x) = \mathcal{P}(u_h|_{K_c}(x), x), \qquad x \in K_c.$$
- **Global** <span style="color: red;">(Non implémenté)</span> : la valeur post-processée en un point $x$ dépend de la valeur reconstruite $u_h$ sur l'ensemble du domaine $\Omega$, c'est-à-dire que la solution post-processée s'écrit :
$$\bar{u_h}(x) = \mathcal{P}(u_h(x), x), \qquad x \in \Omega.$$

### Enregistrement comme pytree JAX

`LocalPostProcessing` est enregistré comme pytree JAX avec :
- `children` : l'opérateur `op` uniquement lorsqu'il s'agit d'un module `eqx.Module` (cas apprenable)
- `aux_data` : `local`, `linear`, `input_dim`, `output_dim`, et `op` lorsqu'il est statique (callable analytique)