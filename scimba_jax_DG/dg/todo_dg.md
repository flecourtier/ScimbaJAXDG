# <span style="color:blue"><b>TODO :</b></span> Classe `EllipticDGscheme`

## **Notations**

**Généralités :**

- $d=1$ : dimension physique
- $n_c$ : nombre de cellules
- $n_{\text{quad}}$ : nombre de points de quadrature
- $n_b$ : ordre de la base (`order`)
- $n_u$ : nombre total de variables (`out_dim`)

**Spécifique à ScimBa :** (mapping, post-processing, bases et/ou terme source apprenables)

- $\theta$ : paramètres du réseau de neurones (notation générique pour le mapping, le post-processing et les bases apprenables)
- $g, g_\theta$ : mapping du maillage (composition de $m$ fonctions) $\longrightarrow$ analytique, réseau de neurones
- $\mathcal{P}, \mathcal{P}_\theta$ : post-processing $\longrightarrow$ analytique, réseau de neurones
- $\varphi_{k,i}$, $\varphi_{k,i}^{\theta,P}$, $\varphi_{k,i}^{\theta,C}$ : la $i$-ème fonction de base (trial) dans la $k$-ème cellule $\longrightarrow$ Taylor, un réseau de neurones (`Patchwise`), $n_c$ réseaux de neurones (`Cellwise`)
- $f$, $f_\theta$ : terme source $\longrightarrow$ analytique, réseau de neurones

## **Cas tests**

- [x] **Cas test 1 (`assemble_volume_term.py`) :**

    - *Tester :* l'assemblage du terme volumique pour un scalaire ($n_u=1$) avec une base analytique (Taylor).

    - *Détails :* Prendre un cas test analytique et vérifier que l'assemblage est bon. (Par exemple les dofs à 1 pour $u$ et $f(x) = \sin(\pi x)$).

    - *Valider :* `assembly_local_volume_term` (forme bilinéaire $a$ et forme linéaire $l$). 

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases |
    |---------|--------|-------------|---------|-------------|---------|--------|
    | 100 | 3 | 2 | 1 | $\sin(\pi x)$ | / | $\varphi_{k,i}$ |

- [x] **Cas test 2 (`assemble_volume_term_system.py`) :**

    - *Tester :* l'assemblage du terme volumique pour un système ($n_u=2$) avec une base analytique (Taylor).

    - *Détails :* Définir `SystemLaplacianWeakForm` héritant de `AbstractLinearWeakForm`, avec une forme bilinéaire et linéaire vectorielles. Utiliser `ParamVecFunction` et ses composantes (`u_1`, `u_2`) pour exprimer la forme faible.

    - *Valider :* `assembly_local_volume_term` dans le cas vectoriel (`basis_type="vec"`).

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases |
    |---------|--------|-------------|---------|-------------|---------|--------|
    | 100 | 3 | 2 | 2 | $(\sin(\pi x),\, \sin(2\pi x))$ | / | $\varphi_{k,i}$ |

- [x] **Cas test 3 (`assemble_volume_term_learnable_source.py`) :**

    - *Tester :* l'assemblage du terme volumique avec un terme source apprenable ($f_\theta$), une base `Patchwise` apprenable et un mapping apprenable.

    - *Détails :* Utiliser `LaplacianWeakFormLearnableSource` avec un `SourceNN` (MLP). Définir une `BasisNN` (MLP) pour la base `Patchwise` et une `MappingNN` (InvertibleNet) pour le mapping. 

    - *Valider :* `assembly_local_volume_term` avec bases et mapping apprenables + différentiation automatique JAX à travers l'assembleur.

    ---

    | $n_c$ | $n_\text{quad}$ | $n_b$ | $n_u$ | $f(x)$ | Mapping | Bases |
    |---------|--------|-------------|---------|-------------|---------|--------|
    | 1 | 3 | 2 | 1 | $f_\theta(x)$ | $g_\theta$ | $\varphi_{k,i}^{\theta,P}$ |


- [ ] **Supplémentaires :**

    - [x] Rajouter un test sur les quadratures pour vérifier les calculs d'intégrales.
    - [ ] Faire la documentation (en anglais).
    - [x] Passer les fonctions qui dépendent de `self` en `@staticmethod`.