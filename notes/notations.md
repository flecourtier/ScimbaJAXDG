# Notations

## Générales

- $\Omega$ : domaine physique
- $d$ : dimension spatiale
- $\ell$ : indice de direction spatiale ($\ell \in \{0, \ldots, d-1\}$)
- $\theta$ : paramètres du/des réseau.x de neurones (notation générique pour le mapping, le post-processing et les bases apprenables)

## Mapping

- $g, g_\theta$ : mapping du maillage (composition de $m$ fonctions), respectivement : analytique ou réseau de neurones
- $m$ : nombre de fonctions composées dans le mapping

## Quadrature

- $q$ : ordre de quadrature par direction
- $n_\text{quad} = q^d$ : nombre total de points de quadrature volumique
- $n_\text{quad}^s = q^{d-1}$ : nombre total de points de quadrature surfacique

## Maillage

- $\mathcal{T}_h$ : maillage uniforme du domaine logique $[0,1]^d$
- $h$ : taille caractéristique du maillage <span style="color: red;">-> à définir</span>

## Cellules

- $N_\ell$ : nombre de cellules dans la direction $\ell$
- $N_\text{cells} = (N_0, \ldots, N_{d-1})$ : tuple du nombre de cellules par direction (`n_cells`)
- $n_\text{cells} = \prod_{\ell=0}^{d-1} N_\ell$ : nombre total de cellules (`n_cells_total`)
- $K_\text{ref} = [0,1]^d$ : élément (cellule) de référence
- $K_c$ : la $c$-ème cellule du maillage,
    - $c \in [0, n_\text{cells} - 1]$ : indexation plate
    - $\bar{c}(c) = (\bar{c}_0, \ldots, \bar{c}_{d-1})$ : indexation multi-dimensionnelle de la cellule $K_c$ avec $\bar{c}_\ell \in \{0, \ldots, N_\ell - 1\}$
- $T_{K_c}$ : transformation géométrique de l'élément de référence vers une cellule $K_c$
- $\Phi_{K_c} = g \circ T_{K_c}$ : composition du mapping et de la transformation géométrique, passage de l'élément de référence au domaine physique

## Faces

- $n_\text{faces}$ : nombre total de faces (`n_faces`)
- $n_{f,\ell}$ : nombre de faces du groupe $\ell$ (normales à la direction $\ell$)
- $F_{\text{ref},\ell} \subset [0,1]^{d}$ : face de référence (hyperplan de $K_\text{ref}$) associé à la direction $\ell$
- $F_k$ : la $k$-ème face du maillage,
    - $k \in [0, n_\text{faces} - 1]$ : indexation plate
    - $\bar{k}(k) = (\ell; \bar{k}_0, \ldots, \bar{k}_{d-1})$ : indexation multi-dimensionnelle pour une face $F_k$ dans le groupe $\ell$ avec 
    $$\bar{k}_\ell \in \{0,\ldots,N_\ell\}, \quad \bar{k}_j \in \{0,\ldots,N_j-1\} \text{ pour } j \in \{0, \ldots, d-1\} \setminus \{\ell\}$$
- $T_{F_k}$ : transformation géométrique de la face de référence vers une face $F_k$ (associée au groupe $\ell$)
- $\Phi_{F_k} = g \circ T_{F_k}$ : composition du mapping et de la transformation géométrique, passage de la face de référence au domaine physique

## Points et transformations géométriques

Tous les points de quadrature (volumique ou surfacique) sont définis dans un espace $\mathbb{R}^d$ (même pour les points surfaciques, où une coordonnée est fixée à $0$).

- $\hat{\xi}$, $\xi$, $X$ : points dans les espaces de référence, logique et physique (respectivement)
- $\hat{\xi}_i$ : $i$-ème point de quadrature sur l'élément de référence $K_\text{ref}$
- $w_i$ : $i$-ème poids de quadrature sur l'élément de référence $K_\text{ref}$
- $\hat{\xi}_{\ell,i}^s$ : $i$-ème point de quadrature surfacique sur la face de référence $F_{\text{ref},\ell}$ (l-ème coordonnée fixée à $0$)
- $w_i^s$ : $i$-ème poids de quadrature surfacique sur la face de référence $F_{\text{ref},\ell}$ (indépendant de $\ell$, car identique pour toutes les faces de référence) <span style="color: red;">-> à vérifier</span>
- $w_{c,i}$, $w_{k,i}^s$ : poids corrigés pour une cellule $K_c$ et une face $F_k$ (respectivement) 

## Bases

- $n_b$ : nombre de fonctions de base par cellule (`nb_basis`)
- $n_{\text{out}}$ : dimension de sortie, nombre de variables dans le système (`out_dim`) 
    <span style="color: red;">-> à clarifier</span>
- $\bar{\varphi}_{c,i}, \bar{\psi}_{c,i}$ : notations génériques pour la $i$-ème fonction de base de la $c$-ème cellule, respectivement : trial et test
- $\bar{\varphi}_{c,i,\alpha}, \bar{\psi}_{c,i,\alpha}$ : $i$-ème fonction de base de la $c$-ème cellule associée à la composante $\alpha$-ème de la solution, respectivement : trial et test
- $\varphi_{c,i}$, $\varphi_{c,i}^{\theta,P}$, $\varphi_{c,i}^{\theta,C}$ : la $i$-ème fonction de base (trial) dans la $c$-ème cellule, respectivement : analytique, patchwise (un réseau de neurones), cellwise ($n_\text{cells}$ réseaux de neurones)

## Variables discrètes

- $U_h : \Omega \to \mathbb{R}^{n_\text{out}}$ : variable discrète reconstruite à partir des DOFs linéaires et des fonctions de base
- $U_{h,\alpha}$ : $\alpha$-ème composante de la variable discrète $U_h$ (pour $\alpha \in \{0, \ldots, n_\text{out}-1\}$)

### Spécifique à la méthode DG

On définit également les notations suivantes pour l'implémentation spécifique à la méthode DG :

- $u_{c,i,\alpha}$ : DOF linéaire associé à la $i$-ème fonction de base de la $c$-ème cellule et à la composante $\alpha$-ème de la solution
- $\mathbf{u}_{c,\alpha} = (u_{c,i,\alpha})_{i=0}^{n_b-1}$ : vecteur des DOFs linéaires associés à la composante $\alpha$-ème de la solution dans la cellule $c$

### Spécifique à la méthode FEM

TODO

## Post-processing

- $\mathcal{P}, \mathcal{P}_\theta$ : post-processing, respectivement : analytique, réseau de neurones
- $\bar{U_h}$ : variable discrète post-processée
<!-- - $\bar{U}_{h,\alpha}$ : $\alpha$-ème composante de la variable discrète post-processée $\bar{U_h}$ (pour $\alpha \in \{0, \ldots, n_\text{out}-1\}$) -->