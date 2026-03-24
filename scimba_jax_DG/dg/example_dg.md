# Tests DG expliqués (dans l'ordre)

Ce document résume les vérifications mathématiques réalisées dans les trois scripts:

1. `assemble_volume_terms`
2. `assemble_flux_term`
3. `assemble_scheme`

Les scripts `assemble_volume_terms_system` et `assemble_volume_terms_learnable_source`, qui traitent des cas système ($n_u>1$) et source learnable, sont similaires à `assemble_volume_terms` et ne sont pas décrits en détail ici, mais les mêmes types de vérifications y sont effectués.

## Notations et similitudes

- $d=1$ : dimension physique
- $n_c=100$ : nombre de cellules
- $n_f=n_c+1$ : nombre de faces (internes + bords)
- $h=1/n_c$ : taille de cellule
- $n_{\text{quad}}=3$ : nombre de points de quadrature
- $n_b=2$ : nombre de fonctions de base locales (ordre 2)
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

## 1) `assemble_volume_terms`

Objectif: vérifier les termes volumiques (bilinéaire et linéaire) cellule par cellule.

Le script appelle `assembly_local_volume_terms(idx, j)`, puis vectorise avec `vmap` + `jit` pour obtenir la contribution bilinéaire `b_integrals` de shape $(n_c, n_b, n_u)$ et la contribution linéaire `l_integrals` de shape $(n_c, n_b, n_u)$. 

La formulation faible locale dans la cellule $C_k$ est:

$$\underbrace{\int_{C_k} \nabla u_h(x) \nabla\varphi_{k,j}(x)\,dx}_{b[k,j,0]} = \underbrace{\int_{C_k} f(x) \varphi_{k,j}(x)\,dx}_{l[k,j,0]} \quad \forall j \in \{0, \ldots, n_b - 1\}$$

**Assertions:**

- $b[:,0,0] \approx 0$

  $\longrightarrow$ $\nabla\varphi_{k,0}=0$, donc pour $j=0$ la partie bilinéaire s'annule

- $b[:,1,0] \approx h$
  
  $\longrightarrow$ $\nabla u_h = 1$ et $\nabla \varphi_{k,1} = 1$ donc pour $j=1$ :

  $$\int_{C_k} \nabla u_h \nabla \varphi_{k,1}\,dx \approx \int_{C_k} 1\,dx = h.$$

**Vérifications visuelles:**

- $l[k,0,0]-f(\bar{x_k})\,h \approx 0$

  $\longrightarrow$ la partie linéaire doit approcher la moyenne de $f$ sur la cellule, qui est proche de la valeur de $f$ au centre pour une fonction lisse.

## 2) `assemble_flux_term`

Objectif: vérifier les contributions de flux numérique DG sur les faces internes.

Le script appelle `assembly_local_flux_term(idf, j)`, puis vectorise avec `vmap` + `jit` pour obtenir:

- `idxL`, `idxR` de shape $(n_f-2, n_b)$ (indices des cellules gauche/droite associées à chaque face interne);
- `fluxL`, `fluxR` de shape $(n_f-2, n_b, n_u)$ (contributions de flux à injecter dans les résidus locaux gauche/droite).

Dans cet exemple, on considère un flux numérique centré(`CenteredFlux`) héritant de la classe `AbstractFlux`.

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

Avec la convention de signe utilisée dans le code:

- `fluxL = -summed_flux = +1`;
- `fluxR =  summed_flux = -1`.

**Assertions:**

- `fluxL[:,:,0] \approx 1`

  $\longrightarrow$ la contribution côté gauche est constante et positive sur toutes les faces internes.

- `fluxR[:,:,0] \approx -1`

  $\longrightarrow$ la contribution côté droit est constante et opposée, ce qui valide la cohérence d'orientation gauche/droite.

<!-- ## 3) `assemble_scheme`

Objectif: vérifier l'assemblage complet du résidu DG (volume + flux).

## Vérification de structure globale

Le résidu assemblé `res` doit avoir la forme:

$$
(n_c\times n_b\times n_u,)
$$

puis est réindexé en grille:

$$
res\_grid \in \mathbb{R}^{n_c\times n_b\times n_u}.
$$

## Test intérieur (cellules 1 à n_c-2)

Analyse attendue:

- Terme volume:
  - $j=0$: $res_{vol}\approx -f(x_c)\,h$
  - $j=1$: $res_{vol}\approx h$
- Terme flux (faces internes): contributions opposées qui se compensent dans une cellule intérieure.

Donc:

$$
res[k,0,0] \approx -f(x_{c,k})\,h,
\qquad
res[k,1,0] \approx h.
$$

Ces deux relations sont vérifiées par assertions `allclose`.

## Test bord (cellules 0 et n_c-1)

Aux bords, la compensation des flux n'est pas symétrique comme à l'intérieur:

- la cellule gauche reçoit un décalage net $+1$;
- la cellule droite reçoit un décalage net $-1$.

Le script vérifie donc:

$$
res[0,j]=volume[0,j]+1,
\qquad
res[n_c-1,j]=volume[n_c-1,j]-1
$$

pour $j=0$ et $j=1$.

## Résumé rapide

- `assemble_volume_terms`: valide la partie volumique locale (formes + valeurs attendues de la bilinéaire, contrôle du linéaire).
- `assemble_flux_term`: valide la partie flux locale sur faces internes (formes + signes + valeur analytique constante).
- `assemble_scheme`: valide la combinaison des deux au niveau du résidu global (intérieur et bords). -->
