# 1 Solve Laplacian [(voir)](images/solve_laplacian_compare_flux.png)

Référence pour les formulations : [Unified Analysis of Discontinuous Galerkin Methods for Elliptic Problems. Douglas N. Arnold, Franco Brezzi, Bernardo Cockburn, and L. Donatella Marini](https://www-users.cse.umn.edu/~arnold/papers/dgerr.pdf)
## 1.1 Formulation primale

### 1.1.1 Le problème modèle

On considère le problème de Poisson avec conditions de Dirichlet homogènes :

$$-\Delta u = f \text{ dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

où $\Omega$ est un polygone convexe et $f \in L^2(\Omega)$.

*Exemple :* 
$$u_\text{ex}(x) = \frac{1}{\pi^2}\sin(\pi x), \quad f(x) = \sin(\pi x)$$
### 1.1.2 Reformulation en système du premier ordre

On introduit la variable auxiliaire $\sigma = \nabla u$, ce qui donne le système :

$$\sigma = \nabla u, \qquad -\nabla \cdot \sigma = f \text{ dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

L'intérêt est de ramener le problème d'ordre 2 en un système d'ordre 1, plus facile à discrétiser avec des éléments discontinus.

### 1.1.3 Formulation faible sur un élément $K$

On multiplie les deux équations par des fonctions test $\tau$ et $v$, et on intègre sur un élément $K$ du maillage $\mathcal{T}_h$. En intégrant par parties :

$$\int_K \sigma \cdot \tau \, dx = -\int_K u \, \nabla \cdot \tau \, dx + \int_{\partial K} u \, n_K \cdot \tau \, ds$$

$$\int_K \sigma \cdot \nabla v \, dx = \int_K f v \, dx + \int_{\partial K} \sigma \cdot n_K \, v \, ds$$

### 1.1.4 Les espaces d'approximation $V_h$ et $\Sigma_h$
 
On se donne un maillage $\mathcal{T}_h = \{K_k\}_{k=0}^{n_c-1}$ de $\Omega$. On note $p \geq 1$ le degré polynomial et on définit les espaces d'éléments finis **discontinus** :
 
$$V_h := \{ v \in L^2(\Omega) : v|_K \in \mathbb{P}_p(K) \quad \forall K \in \mathcal{T}_h \}$$
 
$$\Sigma_h := \{ \tau \in [L^2(\Omega)]^2 : \tau|_K \in [\mathbb{P}_p(K)]^2 \quad \forall K \in \mathcal{T}_h \}$$
 
où $\mathbb{P}_p(K)$ est l'espace des polynômes de degré au plus $p$ sur $K$. La différence essentielle avec les éléments finis classiques est que **les fonctions de $V_h$ et $\Sigma_h$ n'ont aucune contrainte de continuité aux interfaces** entre éléments — elles appartiennent simplement à $L^2(\Omega)$, pas à $H^1(\Omega)$.

### 1.1.5 Le passage aux flux numériques

Dans une méthode DG, les fonctions $u_h \in V_h$ et $\sigma_h \in \Sigma_h$ sont **discontinues aux interfaces**. Les traces sur $\partial K$ sont donc ambiguës : il y a deux valeurs possibles de chaque côté d'une interface.

On remplace les traces exactes $u|_{\partial K}$ et $\sigma|_{\partial K}$ par des **flux numériques** $\hat{u}_K$ et $\hat{\sigma}_K$, construits à partir des valeurs des deux côtés. On cherche alors $u_h \in V_h$ et $\sigma_h \in \Sigma_h$ tels que pour tout $K \in \mathcal{T}_h$ :

$$\int_K \sigma_h \cdot \tau \, dx = -\int_K u_h \, \nabla \cdot \tau \, dx + \int_{\partial K} \hat{u}_K \, n_K \cdot \tau \, ds \qquad \forall \tau \in \Sigma(K)$$

$$\int_K \sigma_h \cdot \nabla v \, dx = \int_K f v \, dx + \int_{\partial K} \hat{\sigma}_K \cdot n_K \, v \, ds \qquad \forall v \in P(K)$$

Le choix des flux $\hat{u}_K$ et $\hat{\sigma}_K$ en fonction de $u_h$ et $\sigma_h$ définit entièrement la méthode DG.

### 1.1.6 Sommation sur tous les éléments

On somme les formules sur tous les éléments $K \in \mathcal{T}_h$.

$$\int_\Omega \sigma_h \cdot \tau \, dx = -\int_\Omega u_h \, \nabla_h \cdot \tau \, dx + \sum_{K\in\mathcal{T}_h} \int_{\partial K} \hat{u}_K \, n_K \cdot \tau \, ds$$

$$\int_\Omega \sigma_h \cdot \nabla_h v \, dx = \int_\Omega f v \, dx + \sum_{K\in\mathcal{T}_h} \int_{\partial K} \hat{\sigma}_K \cdot n_K \, v \, ds$$

et on remarque que les sommes de bord $\sum_K \int_{\partial K}$ font apparaître chaque interface deux fois. 

### 1.1.7 Définition des termes de saut et de moyenne

On définit les opérateurs de saut $[\![\cdot]\!]$ et de moyenne $\{\cdot\}$ sur les interfaces intérieures du maillage :

- Pour une fonction **scalaire** $q$ sur une interface intérieure $e$ partagée par deux éléments $K^+$ et $K^-$ de normales sortantes $n^+$ et $n^-$ :

    $$\{q\} = \frac{1}{2}(q^+ + q^-) \qquad [\![q]\!] = q^+ n^+ + q^- n^-$$

    La moyenne est un **scalaire**, le saut est un **vecteur** (parallèle à la normale).

- Pour une fonction **vectorielle** $\varphi$ :

    $$\{\varphi\} = \frac{1}{2}(\varphi^+ + \varphi^-) \qquad [\![\varphi]\!] = \varphi^+ \cdot n^+ + \varphi^- \cdot n^-$$

    La moyenne est un **vecteur**, le saut est un **scalaire**.

Sur les bords $e \in \partial\Omega$, on pose par convention :

$$[\![q]\!] = q\, n \qquad \{\varphi\} = \varphi$$

où $n$ est la normale sortante (du domaine).

### 1.1.8 Réécriture des termes de bord avec les opérateurs de saut et de moyenne

On réécrit alors les sommes de bord à l'aide de l'identité suivante :

$$\sum_{K\in\mathcal{T}_h} \int_{\partial K} q_K \, \varphi_K \cdot n_K \, ds = \int_\Gamma [\![q]\!] \cdot \{\varphi\} \, ds + \int_{\Gamma_I} \{q\} \, [\![\varphi]\!] \, ds \tag{1}$$

où $\Gamma$ est l'union de tous les bords d'éléments (intérieurs + bords), $\Gamma_I = \Gamma \setminus \partial\Omega$ est l'ensemble des interfaces intérieures.

### 1.1.9 Formulation globale après sommation
 
Après avoir sommé sur tous les éléments et réécrit les termes de bord avec (1), on obtient deux équations globales :
 
$$\int_\Omega \sigma_h \cdot \tau \, dx = -\int_\Omega u_h \nabla_h \cdot \tau \, dx + \int_\Gamma [\![\hat{u}]\!] \cdot \{\tau\} \, ds + \int_{\Gamma_I} \{\hat{u}\}\, [\![\tau]\!] \, ds \tag{2}$$
 
$$\int_\Omega \sigma_h \cdot \nabla_h v \, dx - \int_\Gamma \{\hat{\sigma}\} \cdot [\![v]\!] \, ds - \int_{\Gamma_I} [\![\hat{\sigma}]\!]\{v\} \, ds = \int_\Omega f v \, dx \tag{3}$$
 
L'objectif est maintenant d'éliminer $\sigma_h$ de ce système pour obtenir une équation ne portant que sur $u_h$.
 
### 1.1.10 Expression de $\sigma_h$ en fonction de $u_h$
 
Pour transformer le terme $-\int_\Omega u_h \nabla_h \cdot \tau \, dx$ dans (2), on utilise l'intégration par parties globale suivante (qui est une conséquence de l'identité (1)) :
 
$$-\int_\Omega u_h \nabla_h \cdot \tau \, dx = \int_\Omega \nabla_h u_h \cdot \tau \, dx - \int_\Gamma \{\tau\} \cdot [\![u_h]\!] \, ds - \int_{\Gamma_I} [\![\tau]\!]\{u_h\} \, ds$$
 
En substituant dans (2), les termes en $\{\tau\}$ et $[\![\tau]\!]$ se regroupent et font apparaître l'écart entre le flux numérique $\hat{u}$ et la trace de $u_h$ :
 
$$\int_\Omega \sigma_h \cdot \tau \, dx = \int_\Omega \nabla_h u_h \cdot \tau \, dx + \int_\Gamma [\![\hat{u} - u_h]\!] \cdot \{\tau\} \, ds + \int_{\Gamma_I} \{\hat{u} - u_h\}\, [\![\tau]\!] \, ds \tag{4}$$
 
Pour écrire cela de façon compacte, on introduit les **opérateurs de relèvement** $r : [L^2(\Gamma)]^2 \to \Sigma_h$ et $l : L^2(\Gamma_I) \to \Sigma_h$, définis par :
 
$$\int_\Omega r(\varphi) \cdot \tau \, dx = -\int_\Gamma \varphi \cdot \{\tau\} \, ds \qquad \int_\Omega l(q) \cdot \tau \, dx = -\int_{\Gamma_I} q\, [\![\tau]\!] \, ds$$
 
Ces opérateurs relèvent une quantité définie sur les interfaces vers l'espace $\Sigma_h$. Avec cette notation, (4) s'écrit directement comme une expression de $\sigma_h$ en fonction de $u_h$ uniquement :
 
$$\sigma_h = \nabla_h u_h - r([\![\hat{u} - u_h]\!]) - l(\{\hat{u} - u_h\})$$
 
### 1.1.11 La forme bilinéaire primale

Il reste à éliminer $\sigma_h$ de (3). Pour cela, on prend $\tau = \nabla_h v$ dans (4), ce qui exprime $\int_\Omega \sigma_h \cdot \nabla_h v \, dx$ en termes de $u_h$ :
 
$$\int_\Omega \sigma_h \cdot \nabla_h v \, dx = \int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx + \int_\Gamma [\![\hat{u} - u_h]\!] \cdot \{\nabla_h v\} \, ds + \int_{\Gamma_I} \{\hat{u} - u_h\}\, [\![\nabla_h v]\!] \, ds$$
 
En substituant dans (3), on obtient une équation portant uniquement sur $u_h$, que l'on écrit sous la forme $B_h(u_h, v) = \int_\Omega f v \, dx$ avec :
 
$$B_h(u_h, v) := \int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx + \int_\Gamma \Big([\![\hat{u} - u_h]\!] \cdot \{\nabla_h v\} - \{\hat{\sigma}\} \cdot [\![v]\!]\Big) \, ds$$
$$+ \int_{\Gamma_I} \Big(\{\hat{u} - u_h\}[\![\nabla_h v]\!] - [\![\hat{\sigma}]\!]\{v\}\Big) \, ds$$
 
C'est la **formulation primale** de la méthode DG. Toute la richesse des différentes méthodes DG réside dans le choix des flux $\hat{u}$ et $\hat{\sigma}$ : c'est ce choix qui détermine la forme concrète de $B_h$ et donc les propriétés de stabilité, consistance et précision de la méthode.

### 1.1.12 Le problème discret final
 
On peut maintenant écrire le problème discret sous sa forme définitive. On cherche $u_h \in V_h$ tel que :
 
$$B_h(u_h, v) = \int_\Omega f \, v \, dx \qquad \forall v \in V_h$$
 
C'est une équation variationnelle classique : le membre de droite est le terme source $f$ testé contre $v$, exactement comme dans la formulation faible continue. Le membre de gauche $B_h(u_h, v)$ remplace la forme bilinéaire $\int_\Omega \nabla u \cdot \nabla v \, dx$ du problème continu, en y ajoutant tous les termes d'interface qui assurent la communication entre éléments et la stabilité de la méthode.
 

## 1.2 Choix du flux numérique

On s'intéresse à trois flux numériques (SIPG, NIPG et Babuska-Zlaman) dont les propriétés sont résumées dans la table ci-dessous.

![[table_flux.png]]
### 1.2.1 Symmetric Interior Penalty Galerkin (SIPG)

SIPG est l'une des méthodes DG les plus utilisées pour les problèmes elliptiques. Elle est obtenue en choisissant les flux numériques de façon à ce que la forme bilinéaire $B_h$ soit **symétrique**, **consistante**, et **stable** moyennant un paramètre de pénalité suffisamment grand.
 
#### 1.2.1.1 Choix des flux
 
Le flux scalaire $\hat{u}$ est pris comme la **moyenne** de $u_h$ aux interfaces :
 
$$\hat{u} = \{u_h\} \text{ sur } \Gamma_I, \qquad \hat{u} = 0 \text{ sur } \partial\Omega$$
 
Le flux vectoriel $\hat{\sigma}$ est pris comme la moyenne du gradient brisé, auquel on ajoute un **terme de pénalité** sur le saut de $u_h$ :
 
$$\hat{\sigma} = \{\nabla_h u_h\} - \eta \, [\![u_h]\!] \text{ sur } \Gamma_I, \qquad \hat{\sigma} = \nabla_h u_h - \eta \, u_h \, n \text{ sur } \partial\Omega$$
 
où $\eta > 0$ est le **paramètre de pénalité**, défini sur chaque interface $e$ par $\eta|_e = \eta_0 / h_e$, avec $h_e$ la taille de l'interface et $\eta_0 > 0$ une constante à choisir suffisamment grande.
 
Le terme de pénalité $\eta [\![u_h]\!]$ pénalise le saut de $u_h$ aux interfaces : plus $\eta$ est grand, plus on force $u_h$ à être continue au sens faible à travers les interfaces.
 
#### 1.2.1.2 La forme bilinéaire SIPG

En injectant ces choix de flux dans la formulation primale, on obtient :

$$B_h(u_h, v) := \underbrace{\int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx}_{\text{diffusion brisée}} - \underbrace{\int_\Gamma \Big( \{\nabla_h u_h\} \cdot [\![v]\!] + [\![u_h]\!] \cdot \{\nabla_h v\} \Big) \, ds}_{\text{consistance + symétrie}} + \underbrace{\int_\Gamma \eta \, [\![u_h]\!] \cdot [\![v]\!] \, ds}_{\text{pénalité}}$$

Le premier terme est la diffusion brisée, analogue au terme de la formulation continue. Le deuxième assure que la solution exacte satisfait bien la formulation discrète. Le troisième contrôle les sauts aux interfaces et garantit la stabilité.

#### 1.2.1.3 Propriétés
 
- **Consistance.** La solution exacte $u$ satisfait $B_h(u, v) = \int_\Omega f v \, dx$ pour tout $v \in V_h$, ce qui implique l'orthogonalité de Galerkin :
    
    $$B_h(u - u_h, v) = 0 \qquad \forall v \in V_h$$
    
    Cela découle du fait que $[\![u]\!] = 0$ et $\{\nabla u\} = \nabla u$ pour la solution exacte, qui est régulière.
    
- **Symétrie.** La forme $B_h$ est symétrique : $B_h(u_h, v) = B_h(v, u_h)$. C'est la conséquence directe du signe $-$ devant les deux termes de consistance (un pour $u_h$, un pour $v$). Cette propriété est ce qui distingue SIPG de NIPG (*Non-symmetric* IP), où le signe du second terme est inversé.
    
- **Stabilité.** La méthode est stable dès que $\eta_0$ est suffisamment grand, au sens où il existe une constante $C_s > 0$ telle que :
    
    $$B_h(v, v) \geq C_s \, \| v \|_{1,h}^2 \qquad \forall v \in V_h$$
    
    où $\|v\|_{1,h}^2 = \int_\Omega |\nabla_h v|^2 \, dx + \int_\Gamma \eta \, |[\![v]\!]|^2 \, ds$ est la norme d'énergie naturelle de la méthode.
 
#### 1.2.1.4 Paramètre de pénalité $\eta$
 
Le choix $\eta|_e = \eta_0 / h_e$ est standard. En pratique, pour des polynômes de degré $p$ sur des triangles réguliers (ou sur le maillage 1D), une valeur couramment utilisée est $\eta_0 = p(p+1)$, ce qui garantit la stabilité. Une valeur trop petite de $\eta_0$ rend la forme bilinéaire indéfinie ; une valeur trop grande n'affecte pas la consistance mais peut détériorer le conditionnement du système linéaire.
 
#### 1.2.1.5 Convergence
 
Sous les hypothèses de régularité $u \in H^{p+1}(\Omega)$, SIPG atteint les taux de convergence optimaux suivants :
 
| Norme | Taux de convergence |
|---|---|
| $\|\cdot\|_{1,h}$ (énergie) | $\mathcal{O}(h^p)$ |
| $\|\cdot\|_{L^2(\Omega)}$ | $\mathcal{O}(h^{p+1})$ |

#### 1.2.1.6 Implémentation

Le terme de diffusion brisée ainsi que la forme linéaire sont construits dans la méthode `_assembly_local_volume_terms_pure` de `EllipticDGscheme` et dépendent de `pde` (forme bilinéaire et linéaire). Les termes de flux sont gérés par deux méthodes distinctes :

- **Faces intérieures** : `_assembly_local_interior_flux_term_pure` calcule les contributions aux deux cellules voisines `(idxL, fluxL)` et `(idxR, fluxR)` via `flux.__call__`.
- **Faces frontières (conditions de Dirichlet)** : `_assembly_local_boundary_flux_term_pure` prend en argument supplémentaire une fonction `dirichlet_bc` et calcule la contribution à la cellule intérieure via `flux.boundary_call`. Ces termes ne sont assemblés que si `dirichlet_bc` est fourni à `_assembly_scheme_pure`.

C'est dans la classe `SIPGFlux` que le flux numérique est défini, via `__call__` pour les faces intérieures et `boundary_call` pour les faces frontières avec condition de Dirichlet.

### 1.2.2 Non-symmetric Interior Penalty Galerkin (NIPG)

NIPG est une variante directe de SIPG obtenue en **inversant le signe du terme d'adjoint-consistance**. La méthode reste consistante et stable, mais perd la propriété d'adjoint-consistance, ce qui dégrade le taux de convergence en norme $L^2$.

#### 1.2.2.1 Choix des flux

Les flux sont identiques à SIPG :

$$\hat{u} = \{u_h\} \text{ sur } \Gamma_I, \qquad \hat{u} = 0 \text{ sur } \partial\Omega$$

$$\hat{\sigma} = \{\nabla_h u_h\} - \eta \, [\![u_h]\!] \text{ sur } \Gamma_I, \qquad \hat{\sigma} = \nabla_h u_h - \eta \, u_h \, n \text{ sur } \partial\Omega$$

#### 1.2.2.2 La forme bilinéaire NIPG

La seule différence avec SIPG est le signe du terme en $[\![u_h]\!] \cdot \{\nabla_h v\}$ :

$$B_h(u_h, v) := \underbrace{\int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx}_{\text{diffusion brisée}} - \underbrace{\int_\Gamma \{\nabla_h u_h\} \cdot [\![v]\!] \, ds}_{\text{consistance}} + \underbrace{\int_\Gamma [\![u_h]\!] \cdot \{\nabla_h v\} \, ds}_{\text{non-symétrie}} + \underbrace{\int_\Gamma \eta \, [\![u_h]\!] \cdot [\![v]\!] \, ds}_{\text{pénalité}}$$

Le troisième terme (signe $+$) est exactement celui qui, dans SIPG, portait un signe $-$ et assurait la symétrie et l'adjoint-consistance.

#### 1.2.2.3 Propriétés

- **Consistance.** La solution exacte $u$ satisfait $B_h(u, v) = \int_\Omega f v \, dx$ pour tout $v \in V_h$ : le changement de signe ne porte que sur le terme en $v$, pas sur le terme en $u_h$, donc l'orthogonalité de Galerkin est préservée.

- **Non-symétrie.** La forme $B_h$ n'est plus symétrique : $B_h(u_h, v) \neq B_h(v, u_h)$ en général. C'est la conséquence directe du signe $+$ devant $[\![u_h]\!] \cdot \{\nabla_h v\}$ combiné au signe $-$ devant $\{\nabla_h u_h\} \cdot [\![v]\!]$.

- **Perte d'adjoint-consistance.** Le problème adjoint $-\Delta \psi = g$ n'est pas satisfait au sens discret par la forme $B_h(\cdot, \psi)$. Cela bloque l'argument de dualité standard qui permet de gagner un ordre en $L^2$.

- **Stabilité.** La méthode reste stable pour tout $\eta_0 > 0$, contrairement à SIPG qui requiert $\eta_0 > \eta^*$. En effet, la coercivité de $B_h$ tient quel que soit le signe du terme d'adjoint-consistance dès que le terme de pénalité domine.

#### 1.2.2.4 Convergence

| Norme | Taux de convergence |
|---|---|
| $\|\cdot\|_{1,h}$ (énergie) | $\mathcal{O}(h^p)$ |
| $\|\cdot\|_{L^2(\Omega)}$ | $\mathcal{O}(h^p)$ |

La perte d'un ordre en $L^2$ par rapport à SIPG (qui atteint $\mathcal{O}(h^{p+1})$) est la signature directe de l'absence d'adjoint-consistance. En 1D sur maillage uniforme et pour des problèmes très réguliers, on peut observer de la superconvergence pour $p=1$ (taux effectif ≈ 2 au lieu de 1), mais le phénomène disparaît pour $p \geq 2$ : NIPG donne alors $\mathcal{O}(h^2)$ là où SIPG donne $\mathcal{O}(h^3)$.

#### 1.2.2.5 Implémentation

L'implémentation est identique à SIPG à un signe près. On crée une nouvelle classe `NIPGFlux`.

### 1.2.3 Babuška-Zlámal (BZ)

<span style="color:blue"><b>TODO :</b></span> relire cette section (générée par Claude)

La méthode de Babuška-Zlámal est une **méthode de pénalité pure** : l'unique contribution d'interface est une pénalité sur le saut de $u_h$, sans aucun terme de couplage de gradient. Elle est **inconsistante** et **non adjoint-consistante**, mais l'optimalité est récupérée grâce à une **superpénalité**.

#### 1.2.3.1 Choix des flux

Le flux scalaire $\hat{u}$ est pris comme la **trace intérieure unilatérale** (valeur de l'élément $K$ lui-même, pas la moyenne) :

$$\hat{u}_K = u_h|_K \text{ sur } \partial K$$

Le flux vectoriel $\hat{\sigma}$ ne contient **aucun terme de gradient**, uniquement une pénalité sur le saut :

$$\hat{\sigma} = -\mu \, [\![u_h]\!] \text{ sur } \Gamma$$

où $\mu > 0$ est le **paramètre de superpénalité**, défini par $\mu = \eta \, h_e^{-(2p+1)}$ pour des polynômes de degré $p$, avec $\eta > 0$ une constante.

#### 1.2.3.2 La forme bilinéaire BZ

Avec ce choix de flux, les termes d'interface de la formulation primale se réduisent à la seule pénalité. La forme bilinéaire est :

$$B_h(u_h, v) := \underbrace{\int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx}_{\text{diffusion brisée}} + \underbrace{\int_\Gamma \mu \, [\![u_h]\!] \cdot [\![v]\!] \, ds}_{\text{superpénalité}}$$

Il n'y a **aucun terme en $\{\nabla_h u_h\} \cdot [\![v]\!]$ ni en $[\![u_h]\!] \cdot \{\nabla_h v\}$** : ce sont précisément ces termes qui, dans SIPG et NIPG, assurent la consistance et/ou l'adjoint-consistance.

#### 1.2.3.3 Propriétés

- **Inconsistance.** La solution exacte $u$ ne satisfait **pas** $B_h(u, v) = \int_\Omega f v \, dx$ pour tout $v \in V_h$. En substituant $u$ dans la formulation, il reste un résidu de consistance :

    $$B_h(u, v) = \int_\Omega f v \, dx + \int_\Gamma \{\nabla u\} \cdot [\![v]\!] \, ds$$

    Ce terme résiduel ne disparaît pas, car le terme $\{\nabla_h u_h\} \cdot [\![v]\!]$ qui devrait compenser l'intégration par parties est absent.

- **Non adjoint-consistance.** Par le même raisonnement appliqué au problème adjoint, la forme $B_h(\cdot, \psi)$ ne satisfait pas le problème adjoint. La méthode est donc doublement non-consistante.

- **Stabilité avec superpénalité.** En choisissant $\mu = \eta \, h_e^{-(2p+1)}$, le terme de pénalité domine le résidu de consistance et la méthode devient stable. La condition $\eta > 0$ suffit (pas de seuil sur $\eta$, contrairement à SIPG).

- **Symétrie.** La forme $B_h$ est **symétrique** : $B_h(u_h, v) = B_h(v, u_h)$, car seuls les termes $\int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx$ et $\int_\Gamma \mu \, [\![u_h]\!] \cdot [\![v]\!] \, ds$ sont présents, et les deux sont symétriques.

#### 1.2.3.4 Convergence

Malgré l'inconsistance, la superpénalité permet de retrouver les taux optimaux :

| Norme | Taux de convergence |
|---|---|
| $\|\cdot\|_{1,h}$ (énergie) | $\mathcal{O}(h^p)$ |
| $\|\cdot\|_{L^2(\Omega)}$ | $\mathcal{O}(h^{p+1})$ |

En contrepartie, la superpénalité $\mu \sim h^{-(2p+1)}$ tend à rendre la méthode **très mal conditionnée** : le nombre de condition du système linéaire croît comme $h^{-(2p+2)}$, bien plus vite que pour SIPG ou NIPG où il croît en $h^{-2}$.

#### 1.2.3.5 Implémentation

La classe `BabuSkaZlamalFlux` ne fait intervenir **aucun terme de gradient** dans `__call__` et `boundary_call`. Le paramètre `mu` encapsule directement la superpénalité $\mu = \eta \, h_e^{-(2p+1)}$, qui doit être calculée et passée à la construction de l'objet. Les contributions sont :

- **Faces intérieures** : `fluxL = mu * jump_u * vL * nL`, `fluxR = mu * jump_u * vR * nR`, avec `jump_u = uL * nL + uR * nR`.
- **Faces frontières** : `flux = mu * (u - g) * n * v * n`, où $g$ est la valeur de Dirichlet.

# 2 Solve Diffusion [(voir)](images/solve_diffusion_compare_flux.png)

Le problème considéré est :

$$-\nabla \cdot (A(x) \, \nabla u) = f \quad \text{dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

avec $A$ la matrice de diffusion (propriétés ?). 

*Exemple :* 
$$A(x) = (1 + x) \, I \quad \text{sur} \quad \Omega = [0, 1]$$
$$u_\text{ex}(x) = \sin(\pi x), \quad f(x) = -\pi\cos(\pi x) + \pi^2(1+x)\sin(\pi x)$$

Le Laplacien est le cas particulier $A = I$. La généralisation à $A$ variable ne change **rien à la structure de la méthode DG** : les flux et la formulation primale restent identiques, à ceci près que $\nabla_h u_h$ est remplacé partout par $A \nabla_h u_h$.

## 2.1 Forme bilinéaire et linéaire

La forme volumique change : au lieu de $\int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx$, on intègre la forme bilinéaire de `DiffusionWeakForm` :

$$a(u_h, v) = \int_\Omega (A \nabla_h u_h) \cdot \nabla_h v \, dx$$

Côté implémentation, il suffit de passer `pde = DiffusionWeakForm(dim=1, A=..., f=...)` à la place de `LaplacianWeakForm` : le schéma `EllipticDGscheme` lit automatiquement le champ `A` depuis `pde.fields["A"]` pour assembler le terme de volume.

## 2.2 Flux numériques

Les flux reçoivent `fieldsL[0]` et `fieldsR[0]` qui sont les évaluations de $A$ aux points de quadrature de la face (fournis automatiquement par le schéma). Les classes `SIPGFlux` et `NIPGFlux` utilisent déjà `AL = fieldsL[0]` et `AR = fieldsR[0]` pour pondérer les termes de gradient : `avg_A_grad_u = 0.5 * (AL @ graduL + AR @ graduR)`. **Aucune modification du flux n'est donc nécessaire** : le même objet `SIPGFlux`/`NIPGFlux`/`BabuSkaZlamalFlux` fonctionne pour la diffusion générale.

## 2.3 Paramètre de pénalité

Le choix du paramètre reste le même qu'au Laplacien :

| Flux | Paramètre |
|---|---|
| SIPG / NIPG | `sigma = p*(p+1)`, `h = 1/n_cells` |
| BZ | `mu = 1/h**(2*p+1)` |

# 3 Solve Diffusion/Advection [(voir)](images/solve_diffusion_advection_compare_flux.png)

Le problème considéré est :

$$-\nabla \cdot (A(x) \, \nabla u) + b(x) \cdot \nabla u = f \quad \text{dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

*Exemple :*
$$A(x) = (1 + x) \, I, \quad b(x) = 2 \quad \text{sur} \quad \Omega = [0, 1]$$
$$u_\text{ex}(x) = \sin(\pi x), \quad f(x) = -\pi\cos(\pi x) + \pi^2(1+x)\sin(\pi x) + 2\pi\cos(\pi x)$$

## 3.1 Forme bilinéaire et linéaire

Le terme d'advection $b \cdot \nabla u$ est un terme **purement volumique** — il n'est pas intégré par parties. La forme bilinéaire de `DiffusionAdvectionWeakForm` est :

$$a(u_h, v) = \int_\Omega (A \nabla_h u_h) \cdot \nabla_h v \, dx + \int_\Omega (b \cdot \nabla_h u_h) \, v \, dx$$

Il suffit de passer `pde = DiffusionAdvectionWeakForm(dim=1, A=..., b=..., f=...)`. `DiffusionWeakForm` est un cas particulier avec $b = 0$.

## 3.2 Flux numériques

Le terme d'advection ne génère **aucun terme d'interface** (pas d'intégration par parties). Les flux SIPG/NIPG/BZ restent donc inchangés. L'advection est entièrement prise en charge par le terme de volume.

**Remarque :** (<span style="color:blue"><b>TODO :</b></span> vérifier car générer par Claude) cette formulation sans stabilisation upwind aux interfaces peut produire un régime pré-asymptotique non-monotone en L² pour p=1 lorsque le nombre de Péclet $\mathrm{Pe} = b \, h / A$ n'est pas encore petit. Les taux asymptotiques restent ceux de la diffusion pure.

## 3.3 Paramètre de pénalité

Identique à la diffusion (section 2.3).

# 4 Solve Diffusion/Advection/Reaction [(voir)](images/solve_diffusion_advection_reaction_compare_flux.png)

Le problème considéré est :

$$-\nabla \cdot (A(x) \, \nabla u) + b(x) \cdot \nabla u + c(x) \, u = f \quad \text{dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

*Exemple :*
$$A(x) = (1 + x) \, I, \quad b(x) = 2, \quad c(x) = 1 \quad \text{sur} \quad \Omega = [0, 1]$$
$$u_\text{ex}(x) = \sin(\pi x), \quad f(x) = \pi^2(1+x)\sin(\pi x) - \pi\cos(\pi x) + 2\pi\cos(\pi x) + \sin(\pi x)$$

## 4.1 Forme bilinéaire et linéaire

Le terme de réaction $c \, u$ est également un terme **purement volumique**. La forme bilinéaire de `DiffusionAdvectionReactionWeakForm` est :

$$a(u_h, v) = \int_\Omega (A \nabla_h u_h) \cdot \nabla_h v \, dx + \int_\Omega (b \cdot \nabla_h u_h) \, v \, dx + \int_\Omega c \, u_h \, v \, dx$$

Il suffit de passer `pde = DiffusionAdvectionReactionWeakForm(dim=1, A=..., b=..., c=..., f=...)`. `DiffusionAdvectionWeakForm` est un cas particulier avec $c = 0$.

## 4.2 Flux numériques

Ni l'advection ni la réaction ne génèrent de terme d'interface. Les flux SIPG/NIPG/BZ restent donc inchangés (identiques au cas diffusion pure).

## 4.3 Paramètre de pénalité

Identique à la diffusion (section 2.3).
