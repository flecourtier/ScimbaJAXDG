# 1 Assemblage DG - Présentation des exemples

Ce document résume les vérifications mathématiques réalisées dans les quatres scripts:

1. `assemble_volume_terms`
2. `assemble_centered_flux_term` + `assemble_laplacian_SIPG_flux_term`
3. `assemble_scheme`

Les scripts `assemble_volume_terms_system` et `assemble_volume_terms_learnable_source`, qui traitent des cas système ($n_u>1$) et source learnable, sont similaires à `assemble_volume_terms` et ne sont pas décrits en détail ici, mais les mêmes types de vérifications y sont effectués.

## 1.1 Notations et similitudes

- $d=1$ : dimension physique
- $n_c=100$ : nombre de cellules
- $n_f=n_c+1$ : nombre de faces (internes + bords)
- $h=1/n_c$ : taille de cellule
- $n_{\text{quad}}=3$ : nombre de points de quadrature
- $n_b=2$ : nombre de fonctions de base locales (ordre du polynöme + 1)
- $n_u=1$ : nombre de variables (`out_dim`)
- $\varphi_{k,i}$ : $i$-ème fonction de base de Taylor dans la cellule $k$
- $f$ : terme source analytique
- $u_{k,i}$ : degré de liberté associé à $\varphi_{k,i}$ dans la cellule $k$ (stocké dans `dofsl[k,i,0]`)
- $u_h$ : solution approchée reconstruite à partir des degrés de liberté `dofsl`

On considère un cadre identique dans les trois cas:
- un maillage 1D uniforme sur $[0,1]$ (mapping identité);
- le problème de Poisson (classe `LaplacianWeakForm`) avec conditions de Dirichlet homogènes;
- des bases de Taylor d'ordre 2:
  $$
  \varphi_{k,0}=1 \quad \text{et} \quad \varphi_{k,1}=x-\bar{x_{k}}
  $$
  avec $\bar{x_{k}}$ le centre de la cellule $C_k$;
- un seul champ scalaire ($n_u=1$);
- le vecteur `dofsl` initialisé à 1;
- le terme source $f(x)=\sin(\pi x)$.

Ainsi, la solution reconstruite dans la cellule $C_k$ est:
$$
u_h(x) = (1 + x-\bar{x_{k}}) \mathbb{1}_{\{x\in C_k\}}
$$

## 1.2 Termes de volume : `assemble_volume_terms`

*Objectif: vérifier les termes volumiques (bilinéaire et linéaire) cellule par cellule.*

Le script appelle `assembly_local_volume_terms(dofsl, g)` où `g` est l'indice global définit par $k*n_b+j$ (avec $k$ l'indice de la cellule et $j$ l'indice de la fonction de base). On vectorise avec `vmap` + `jit` pour obtenir la contribution bilinéaire `b_integrals` de shape $(n_c, n_b, n_u)$ et la contribution linéaire `l_integrals` de shape $(n_c, n_b, n_u)$. 

La formulation faible locale dans la cellule $C_k$ est:

$$\underbrace{\int_{C_k} \nabla u_h(x) \nabla\varphi_{k,j}(x)\,dx}_{b[k,j,0]} = \underbrace{\int_{C_k} f(x) \varphi_{k,j}(x)\,dx}_{l[k,j,0]} \quad \forall j \in \{0, \ldots, n_b - 1\}$$

**Assertions:**

- `b[:,0,0]`$\approx 0$

  $\longrightarrow$ $\nabla\varphi_{k,0}=0$, donc pour $j=0$ la partie bilinéaire s'annule

- `b[:,1,0]`$\approx h$
  
  $\longrightarrow$ $\nabla u_h = 1$ et $\nabla \varphi_{k,1} = 1$ donc pour $j=1$ :

  $$\int_{C_k} \nabla u_h \nabla \varphi_{k,1}\,dx \approx \int_{C_k} 1\,dx = h.$$

**Vérifications visuelles:**

- `l[k,0,0]-f(\bar{x_k})\,h \approx 0`

  $\longrightarrow$ la partie linéaire doit approcher la moyenne de $f$ sur la cellule, qui est proche de la valeur de $f$ au centre pour une fonction lisse.

## 1.3 Termes de flux : 

*Objectif: vérifier les contributions de flux numérique DG sur les faces internes.*

*Note: On s'intéresse ici uniquement aux faces internes, les contributions de flux sur les faces de bord sont exclus dans ces exemples.*


Le script appelle `assembly_local_flux_term(dofsl, g)` où `g` est l'indice global définit par $l*n_b+j$ (avec $l$ l'indice de la face interne et $j$ l'indice de la fonction de base). On vectorise avec `vmap` + `jit` pour obtenir:

- `idxL`, `idxR` de shape $(n_f-2, n_b)$ (indices des cellules gauche/droite associées à chaque face interne);
- `fluxL`, `fluxR` de shape $(n_f-2, n_b, n_u)$ (contributions de flux à injecter dans les résidus locaux gauche/droite).

Chaque flux numérique $\mathcal F$ (classe fille de `AbstractFlux`) est alors calculé à partir du `__call__` qui prend en entrée :

- `varL` = `uL`, `vL`, `graduL`, `gradvL`, `nL`, `fieldsL`
- `varR` = `uR`, `vR`, `graduR`, `gradvR`, `nR`, `fieldsR`

**Notation:** On introduit la notation locale de contribution de flux sur une face interne $F_l$:

$$
\Phi_{F_l,j}^{L} \quad \text{et} \quad \Phi_{F_l,j}^{R},
$$

où $\Phi_{F_l,j}^{L}=$`fluxL[l,j,0]` (resp. $\Phi_{F_l,j}^{R}=$`fluxR[l,j,0]`) correspond à la contribution de flux à injecter dans le résidu local de la cellule à gauche (resp. à droite) de $F_l$ pour la fonction de base d'indice $j$.

Autrement dit, `fluxL` et `fluxR` sont les tenseurs qui empilent ces contributions pour tous les indices $(l,j)$.

### 1.3.1`assemble_centered_flux_term`

Dans cet exemple, on considère un flux numérique centré (`CenteredFlux`) héritant de la classe `AbstractFlux`.

La contribution de flux localement assemblée sur une face interne $F_l$ repose donc sur un flux numérique centré:

$$
\mathcal F(u_h^L,u_h^R) = -\frac12\left(u_h^L+u_h^R\right),
$$

où $u_h^L$ (resp. $u_h^R$) est la solution reconstruite à gauche (resp. à droite) de la face $F_l$.

Ainsi, sur une face interne $F_l$, on a:

$$
u_h^L = 1 + \frac{0.5}{n_c},\qquad
u_h^R = 1 - \frac{0.5}{n_c},
$$

et

$$
\mathcal F=-1.
$$

Ainsi, la contribution de flux à injecter dans les résidus locaux gauche/droite est constante et opposée:

$$
\Phi_{F_l,j}^{L}:=-\mathcal F(u_h^L,u_h^R),
\qquad
\Phi_{F_l,j}^{R}:=+\mathcal F(u_h^L,u_h^R),
$$

et

- `fluxL = +1`;
- `fluxR = -1`.

**Assertions:**

- `fluxL[:,:,0] \approx 1`

  $\longrightarrow$ la contribution côté gauche est constante et positive sur toutes les faces internes.

- `fluxR[:,:,0] \approx -1`

  $\longrightarrow$ la contribution côté droit est constante et opposée, ce qui valide la cohérence d'orientation gauche/droite.

### 1.3.2 `assemble_laplacian_SIPG_flux_term`

Dans cet exemple, on considère un flux SIPG (`SIPGFlux`) avec paramètre de pénalité $\sigma$ et taille de maille $h=1/n_c$.

Pour une face interne $F_l$, on simplifit les notations en définissant ses cellules de gauche et de droite respectives par $C_L := C_{l-1}$ et $C_R := C_l$. La normale $n_L$ est orientée vers la droite ($n_L=+1$) et $n_R$ vers la gauche ($n_R=-1$).

Notons $\{u_h\} = \frac12(u_h^L + u_h^R)$ l'opérateur de moyenne et $[u_h] = u_h^L n_L + u_h^R n_R = u_h^L - u_h^R$ le saut orienté, le flux SIPG s'écrit:

$$
\mathcal F((u_h^L,v_h^L,n_L),(u_h^R,v_h^R,n_R)) = -[u_h]\cdot\{\nabla v_h\} - \{\nabla u_h\}\cdot[v_h] + \frac{\sigma}{h}[u_h]\cdot[v_h].
$$

Ainsi, les contributions au résidu local de la cellule de gauche $C_L$ et de la cellule de droite $C_R$ sont respectivement:

$$
\Phi_{F_l,j}^{L} = - \frac12 [u_h]\cdot \nabla \varphi_{k_L,j} -\{\nabla u_h\} \cdot \varphi_{k_L,j} + \frac{\sigma}{h}[u_h]\varphi_{k_L,j},
$$

$$
\Phi_{F_l,j}^{R} = - \frac12 [u_h]\cdot \nabla \varphi_{k_R,j} +\{\nabla u_h\} \cdot \varphi_{k_R,j} - \frac{\sigma}{h}[u_h]\varphi_{k_R,j},
$$

avec $k_L=l-1$ et $k_R=l$ les indices des cellules gauche/droite de la face interne $F_l$.

Avec `dofsl = 1` et la base de Taylor, on a :
$$
\{\nabla u_h\} = 1,\qquad [u_h] = u_h^L - u_h^R = \frac{1}{n_c} = h.
$$

**Pour $j=0$** ($\varphi_{k,0}=1$, $\nabla\varphi_{k,0}=0$):

$$
\Phi_{F_l,0}^{L} = - 0 - 1 \cdot 1 + \frac{\sigma}{h}\cdot h \cdot 1 = \sigma - 1,
$$

$$
\Phi_{F_l,0}^{R} = - 0 + 1 \cdot 1 - \frac{\sigma}{h}\cdot h \cdot 1 = 1 - \sigma.
$$

**Pour $j=1$** ($\varphi_{k,1}=x-\bar x_k$, $\nabla\varphi_{k,1}=1$): 

En considérant $x_f$ le point situé sur la face interne $F_l$, on a :

$$
\varphi_{k_L,1}(x_f) = x_f - \bar x_{k_L} = \frac{h}{2},\qquad \varphi_{k_R,1}(x_f) = x_f - \bar x_{k_R} = -\frac{h}{2},
$$

donc,

$$
\Phi_{F_l,1}^{L} = - \frac12\cdot h\cdot 1 - 1\cdot\frac{h}{2} + \frac{\sigma}{h}\cdot h\cdot\frac{h}{2} = h\!\left(\frac{\sigma}{2}-1\right),
$$

$$
\Phi_{F_l,1}^{R} = - \frac12\cdot h\cdot 1 + 1\cdot\left(-\frac{h}{2}\right) - \frac{\sigma}{h}\cdot h\cdot\left(-\frac{h}{2}\right) = h\!\left(\frac{\sigma}{2}-1\right).
$$

**Assertions** (pour $\sigma=4$, $h=0.01$):

- `fluxL[:,0,0]` $\approx \sigma - 1 = 3$

  $\longrightarrow$ la contribution côté gauche pour $j=0$ est uniforme et vaut $\sigma-1$ sur toutes les faces internes.

- `fluxR[:,0,0]` $\approx 1 - \sigma = -3$

  $\longrightarrow$ la contribution côté droit est l'opposée, ce qui reflète l'antisymétrie du saut.

> **Note numérique:** Dans le code, $u_h^L$ et $u_h^R$ sont évalués en $x_f \pm \varepsilon$ avec $\varepsilon=10^{-6}$, ce qui introduit une erreur $2\sigma n_c \varepsilon = 0.0008$ sur le terme de pénalité. Les assertions utilisent donc `atol=1e-3`.

## 1.4 `assemble_scheme`

Objectif: vérifier l'assemblage complet du résidu DG (volume + flux).

Le script appelle `assembly_scheme()`, puis vectorise/compile avec `jit` pour obtenir le résidu global `res` de shape $(n_c,n_b,n_u)$. Le vecteur est ensuite réindexé en

$$
res\_grid \in \mathbb{R}^{n_c\times n_b\times n_u}
$$

afin d'analyser les contributions cellule par cellule.

La structure du résidu local dans la cellule $C_k$ est

$$
res[k,j,0] = \underbrace{b[k,j,0]-l[k,j,0]}_{\text{volume}} + \underbrace{\Phi_{\partial C_k,j}}_{\text{flux}},
\qquad j\in\{0,\dots,n_b-1\}.
$$

Ici:

- $b[k,j,0]=\int_{C_k} \nabla u_h\,\nabla\varphi_{k,j}\,dx$,
- $l[k,j,0]=\int_{C_k} f\,\varphi_{k,j}\,dx$,
- $\Phi_{\partial C_k,j}$ regroupe les contributions de flux des faces de $C_k$.

En notant $F_l$ la face interne d'indice $l\in\{1,\dots,n_f-2\}$ (avec $F_l$ entre $C_{l-1}$ et $C_l$), on a pour une cellule intérieure $k\in\{1,\dots,n_c-2\}$:

$$
\Phi_{\partial C_k,j}=\Phi_{F_k,j}^{R}+\Phi_{F_{k+1},j}^{L}.
$$

Cela correspond exactement à: contribution depuis la face gauche de $C_k$ (côté droit de $F_k$) + contribution depuis la face droite de $C_k$ (côté gauche de $F_{k+1}$).

Aux cellules de bord, les contributions sur la face de bord elle-même se compensent localement (même cellule des deux côtés de la face dans le code). Le décalage net vient donc de la seule face interne adjacente:

- pour la cellule de bord gauche $C_0$:
$$
\Phi_{\partial C_0,j}=\Phi_{F_1,j}^{L}
$$
- pour la cellule de bord droite $C_{n_c-1}$:
$$
\Phi_{\partial C_{n_c-1},j}=\Phi_{F_{n_c-1},j}^{R}
$$

**Assertions:** 

- **Cellules intérieures:** Soit $k\in\{1,\dots,n_c-2\}$ un indice de cellule intérieure, alors:

  - `res_grid[k,0,0]` $\approx -f(\bar x_k)\,h$

    $\longrightarrow$ pour $j=0$, la partie bilinéaire est nulle ($\nabla\varphi_{k,0}=0$), et les flux internes se compensent, donc il reste essentiellement `-l[k,0,0]` (que l'on a vérifié être proche de $-f(\bar x_k)\,h$).

  - `res_grid[k,1,0]` $\approx h$

    $\longrightarrow$ pour $j=1$, la partie volumique vaut approximativement $h$ et, à nouveau, les contributions de flux internes s'annulent deux à deux.
    Plus précisément, on a $res\_grid[k,1,0]\approx h-l[k,1,0]$ avec
    $l[k,1,0]=\int_{C_k} f(x)(x-\bar x_k)\,dx=O(h^3)$ (symétrie locale autour de $\bar x_k$), donc la partie linéaire est très petite et la valeur reste proche de $h$.

- **Cellules de bords:**

  - `res_grid[0,0,0]` $\approx -f(\bar x_0)\,h + 1$
  - `res_grid[0,1,0]` $\approx h + 1$
  - `res_grid[-1,0,0]` $\approx -f(\bar x_{n_c-1})\,h - 1$
  - `res_grid[-1,1,0]` $\approx h - 1$

  Interprétation:

  - au bord gauche, le résidu reçoit un décalage net $+1$ (provenant de la contribution de flux $\Phi_{F_1,j}^{L}$);
  - au bord droit, le résidu reçoit un décalage net $-1$ (provenant de la contribution de flux $\Phi_{F_{n_c-1},j}^{R}$);
