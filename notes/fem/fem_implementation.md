# Implémentation FEM : Branche `scimba_jax_dg`

<span style="color: blue;">Le fichier "implementation.md" contient toutes les informations communes à l'implémentation du schéma DG et du schéma FE.</span>

<span style="color: blue;">On reprend toutes les notations utilisées dans le fichier `notations.md`.</span>

<span style="color: red;">TODO : Définir les espaces d'approximation.</span>

## Variables

> src/scimba_jax/linear_approximation/variables/variables_fe.py

On considère un système de $n_\text{out}$ équations couplées à résoudre sur un domaine $\Omega \subset \mathbb{R}^d$ (ex : système de Navier-Stokes avec $n_\text{out}=4$ en 2D : vélocité (de taille $d$), pression, température).
On définit la solution discrète $U_h : \Omega \to \mathbb{R}^{n_\text{out}}$ (reconstruite à partir des DOFs linéaires) et $U_{h,\alpha} : \Omega \to \mathbb{R}$ sa $\alpha$-ème composante, $\alpha \in \{0, \ldots, n_\text{out}-1\}$.

En EF, les DOFs sont **nodaux et globaux** : ils sont partagés entre les cellules adjacentes, ce qui assure la continuité de la solution à travers les interfaces. On introduit la **table de connectivité** $\mathcal{C}(c) = (\mathcal{C}(c)_0, \ldots, \mathcal{C}(c)_{n_b-1})$ qui, pour chaque cellule $c$, donne les $n_b$ indices des nœuds globaux lui appartenant.

On dispose de deux écritures équivalentes de cette variable discrète :

- **Écriture locale (cellule par cellule) :**

    $$U_{h,\alpha}|_{g(K_c)}(X) = \sum_{i=0}^{n_b-1} u_{\mathcal{C}(c)_i,\alpha}\, \bar{\varphi}_{c,i,\alpha}(X), \qquad X \in g(K_c).$$

- **Écriture globale :**

    $$U_{h,\alpha}(X) = \sum_{n=0}^{n_\text{nodes}-1} u_{n,\alpha}\, \phi_{n,\alpha}(X), \qquad X \in \Omega.$$

avec $u_{n,\alpha}$ le DOF linéaire associé au $n$-ème nœud global et à la composante $\alpha$-ème de la solution, et $\phi_{n,\alpha}$ la fonction de base globale associée au nœud $n$ et à la composante $\alpha$ (à support compact, typiquement Lagrange).

La différence fondamentale avec le schéma DG est que les DOFs ne sont pas locaux à chaque cellule : le DOF $u_{\mathcal{C}(c)_i,\alpha}$ est partagé par toutes les cellules qui contiennent le nœud $\mathcal{C}(c)_i$.

### Degrés de liberté

Les degrés de liberté linéaires sont stockés dans `dofsl` de shape $(n_\text{nodes}, n_\text{out})$ (un vecteur de valeurs par nœud global), et la table de connectivité dans `connectivity` de shape $(n_\text{cells}, n_b)$.

Pour un maillage $Q_k$ uniforme avec $N_\ell$ cellules dans chaque direction $\ell$, le nombre total de nœuds est $n_\text{nodes} = \prod_\ell (N_\ell \cdot k + 1)$ (les nœuds de bord entre cellules étant partagés).

| Attribut | Valeur |
|---|---|
| `n_nodes_total` | nombre total de nœuds globaux ($n_\text{nodes}$) |
| `connectivity` | table de connectivité de shape $(n_\text{cells}, n_b)$ |
| `ndof_linear` | nombre de DOFs linéaires ($n_\text{nodes} \times n_\text{out}$) |
| `ndof` | nombre de DOFs totaux (incluant les paramètres des réseaux de neurones) |

### Méthodes d'évaluation

| Méthode | Input | Output | Note |
|---|---|---|---|
| `local_evaluate(inputs)` | $(d,)$ | $(n_\text{out},)$ | 1 point (cherche la cellule) |
| `evaluate(inputs)` | $(\cdot, d)$ | $(\cdot, n_\text{out})$ | batch de points arbitraires |
| `evaluate_quad(inputs)` | $(n_\text{cells}, n_{\text{quad}}, d)$ | $(n_\text{cells}, n_{\text{quad}}, n_\text{out})$ | points de quadrature du maillage |

Chaque méthode a une variante `_pure` (méthode statique) qui prend `variables_pytree` comme argument explicite — nécessaire pour la différentiation par rapport aux paramètres.

L'évaluation en un point $X \in g(K_c)$ se fait en deux étapes :

1. Identifier la cellule $c$ contenant $X$ (via `mesh.find_cell_index`).
2. Rassembler les DOFs locaux $\theta_c = (u_{\mathcal{C}(c)_i,\alpha})_{i,\alpha}$ via la table de connectivité, puis évaluer :
$$U_{h,\alpha}(X) = \sum_{i=0}^{n_b-1} \theta_{c,i,\alpha}\, \bar{\varphi}_{c,i,\alpha}(X)$$

### Projection d'une fonction sur l'espace EF

On suppose ici qu'on veut projeter une fonction $f : \Omega \to \mathbb{R}^{n_\text{out}}$ sur l'espace EF.

#### Sans post-processing (classique)

Contrairement au cas DG où le système est résolu localement cellule par cellule, la méthode EF nécessite d'assembler et de résoudre un **système linéaire global**. La continuité entre cellules implique que les DOFs partagés couplent les contributions de toutes les cellules adjacentes.

On cherche $\mathbf{u}_\alpha = (u_{n,\alpha})_{n=0}^{n_\text{nodes}-1}$ tel que :

$$\underbrace{\int_\Omega f(X) \cdot \bar{\psi}_{c,j,\alpha}(X)\, dX}_{(b_\alpha)_j} = \sum_{n=0}^{n_\text{nodes}-1} u_{n,\alpha} \underbrace{\int_\Omega \phi_{n,\alpha}(X) \cdot \bar{\psi}_{c,j,\alpha}(X)\, dX}_{(M_\alpha)_{n,j}}, \quad \forall j,$$

En pratique, la matrice de masse globale $M_\alpha \in \mathbb{R}^{n_\text{nodes} \times n_\text{nodes}}$ et le vecteur global $b_\alpha \in \mathbb{R}^{n_\text{nodes}}$ sont assemblés par accumulation des **contributions locales** de chaque cellule $K_c$ :

$$M_{jj'}^{(c)} = \int_{g(K_c)} \bar{\psi}_{c,j,\alpha}(X) \cdot \bar{\varphi}_{c,j',\alpha}(X)\, dX, \qquad b_j^{(c)} = \int_{g(K_c)} f(X) \cdot \bar{\psi}_{c,j,\alpha}(X)\, dX$$

puis scatterées dans la matrice et le vecteur globaux via la table de connectivité (assemblage classique EF) :

$$(M_\alpha)_{\mathcal{C}(c)_j,\, \mathcal{C}(c)_{j'}} \mathrel{+}= M_{jj'}^{(c)}, \qquad (b_\alpha)_{\mathcal{C}(c)_j} \mathrel{+}= b_j^{(c)}$$

Le système global $M_\alpha\, \mathbf{u}_\alpha = \mathbf{b}_\alpha$ (de taille $n_\text{nodes} \times n_\text{nodes}$) est résolu via `jnp.linalg.solve`.

> **Remarque :** En EF, la matrice de masse globale est **creuse** (chaque ligne ne comporte que des entrées non nulles pour les fonctions de base ayant un support commun). Cependant, pour simplifier l'implémentation en JAX (qui ne gère pas nativement les matrices creuses), on utilise une représentation dense et `jnp.linalg.solve`.

> **Contraste avec DG :** En DG, chaque cellule est indépendante (pas de DOFs partagés), ce qui permet de résoudre $n_\text{cells}$ systèmes $(n_b \times n_b)$ en parallèle via `vmap`. En EF, on résout un unique système global $(n_\text{nodes} \times n_\text{nodes})$.

#### Avec post-processing local (possiblement non-linéaire)

> **Remarques :**
> - Le post-processing local est appliqué à la variable discrète reconstruite $U_h$ (et non pas directement aux DOFs linéaires) : $\bar{U_h}(X) = \mathcal{P}(U_h(X), X)$.
> - Le post-processing peut coupler les différentes composantes de la solution, ce qui implique qu'il est nécessaire de résoudre un système global sur toutes les composantes $\alpha$-ème à la fois.
> - Comme en DG, si le post-processing est non-linéaire, on doit faire un Newton. Si le post-processing est linéaire, une seule itération suffit.
> - Contrairement au cas DG où le Newton est résolu **localement** (un système par cellule via `vmap`), en EF, les DOFs étant partagés entre cellules, le Newton doit être résolu **globalement**.

On cherche $\mathbf{u} = (u_{n,\alpha})_{n,\alpha} \in \mathbb{R}^{n_\text{nodes} \times n_\text{out}}$ tel que :

$$R_{n,\alpha}(\mathbf{u}) := \sum_c \int_{g(K_c)} \left[\mathcal{P}(U_h(X), X) - f(X)\right]_\alpha \cdot \bar{\psi}_{c,j,\alpha}(X)\, dX = 0, \quad \forall n \in \{0,\ldots,n_\text{nodes}-1\},\, \alpha \in \{0,\ldots,n_\text{out}-1\}$$

où la somme porte sur les cellules $c$ telles que $n = \mathcal{C}(c)_j$ pour un certain $j$ (i.e. les cellules contenant le nœud $n$).

En pratique, le résidu global $\mathbf{R} \in \mathbb{R}^{n_\text{nodes} \times n_\text{out}}$ et le Jacobien global $J_\mathbf{R} \in \mathbb{R}^{(n_\text{nodes} \cdot n_\text{out}) \times (n_\text{nodes} \cdot n_\text{out})}$ sont assemblés par accumulation de contributions locales (via `vmap` sur les cellules, puis `jacrev` pour le Jacobien local), puis scatterées dans les tableaux globaux via la table de connectivité.

Le système non-linéaire est résolu par Newton :

$$\mathbf{u}^{(s+1)} = \mathbf{u}^{(s)} - J_\mathbf{R}^{-1}\, \mathbf{R}(\mathbf{u}^{(s)})$$

via `jnp.linalg.lstsq` avec régularisation de Tikhonov ($J_\text{reg} = J_\mathbf{R} + \varepsilon I$). Le VJP est défini via `jax.custom_vjp` (différentiation implicite).

> **Exemple (couplage des composantes) :** Prenons $n_\text{out} = 2$ et un post-processing simple (linéaire) :
>
> $$\bar{U}_h(X) = \mathcal{P}(U_h(X),X) = (U_{h,0}(X) + U_{h,1}(X) , U_{h,1}(X))$$
> Le résidu au nœud $n$, composante $\alpha=0$, est :
> $$R_{n,0}(\mathbf{u}) = \sum_c \sum_{\substack{j \\ \mathcal{C}(c)_j = n}} \int_{g(K_c)} \left[U_{h,0}(X) + U_{h,1}(X) - f_0(X)\right] \cdot \bar{\psi}_{c,j,0}(X)\, dX$$
> Et ainsi $\frac{\partial R_{n,0}}{\partial u_{n',1}} \neq 0$ pour les nœuds $n'$ voisins de $n$, ce qui signifie que le résidu sur la composante $0$ au nœud $n$ dépend des DOFs de la composante $1$ des nœuds voisins via le post-processing $\mathcal{P}$. C'est pourquoi il est nécessaire de résoudre un système global sur toutes les composantes à la fois.
>
> En pratique, si on est dans le cas $\mathcal{P}_\theta$ (un réseau de neurones), toutes les composantes sont mélangées dans les couches cachées, donc tous les blocs hors-diagonale du Jacobien global sont non nuls a priori.
