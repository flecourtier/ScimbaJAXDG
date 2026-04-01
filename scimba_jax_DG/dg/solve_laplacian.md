# Explications - Solve Laplacian

Référence : [Unified Analysis of Discontinuous Galerkin Methods for Elliptic Problems. Douglas N. Arnold, Franco Brezzi, Bernardo Cockburn, and L. Donatella Marini](https://www-users.cse.umn.edu/~arnold/papers/dgerr.pdf)

## Formulation primale

### Le problème modèle

On considère le problème de Poisson avec conditions de Dirichlet homogènes :

$$-\Delta u = f \text{ dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

où $\Omega$ est un polygone convexe et $f \in L^2(\Omega)$.

### Reformulation en système du premier ordre

On introduit la variable auxiliaire $\sigma = \nabla u$, ce qui donne le système :

$$\sigma = \nabla u, \qquad -\nabla \cdot \sigma = f \text{ dans } \Omega, \qquad u = 0 \text{ sur } \partial\Omega$$

L'intérêt est de ramener le problème d'ordre 2 en un système d'ordre 1, plus facile à discrétiser avec des éléments discontinus.

### Formulation faible sur un élément $K$

On multiplie les deux équations par des fonctions test $\tau$ et $v$, et on intègre sur un élément $K$ du maillage $\mathcal{T}_h$. En intégrant par parties :

$$\int_K \sigma \cdot \tau \, dx = -\int_K u \, \nabla \cdot \tau \, dx + \int_{\partial K} u \, n_K \cdot \tau \, ds$$

$$\int_K \sigma \cdot \nabla v \, dx = \int_K f v \, dx + \int_{\partial K} \sigma \cdot n_K \, v \, ds$$

### Les espaces d'approximation $V_h$ et $\Sigma_h$
 
On se donne un maillage $\mathcal{T}_h = \{K_k\}_{k=0}^{n_c-1}$ de $\Omega$. On note $p \geq 1$ le degré polynomial et on définit les espaces d'éléments finis **discontinus** :
 
$$V_h := \{ v \in L^2(\Omega) : v|_K \in \mathbb{P}_p(K) \quad \forall K \in \mathcal{T}_h \}$$
 
$$\Sigma_h := \{ \tau \in [L^2(\Omega)]^2 : \tau|_K \in [\mathbb{P}_p(K)]^2 \quad \forall K \in \mathcal{T}_h \}$$
 
où $\mathbb{P}_p(K)$ est l'espace des polynômes de degré au plus $p$ sur $K$. La différence essentielle avec les éléments finis classiques est que **les fonctions de $V_h$ et $\Sigma_h$ n'ont aucune contrainte de continuité aux interfaces** entre éléments — elles appartiennent simplement à $L^2(\Omega)$, pas à $H^1(\Omega)$.

### Le passage aux flux numériques

Dans une méthode DG, les fonctions $u_h \in V_h$ et $\sigma_h \in \Sigma_h$ sont **discontinues aux interfaces**. Les traces sur $\partial K$ sont donc ambiguës : il y a deux valeurs possibles de chaque côté d'une interface.

On remplace les traces exactes $u|_{\partial K}$ et $\sigma|_{\partial K}$ par des **flux numériques** $\hat{u}_K$ et $\hat{\sigma}_K$, construits à partir des valeurs des deux côtés. On cherche alors $u_h \in V_h$ et $\sigma_h \in \Sigma_h$ tels que pour tout $K \in \mathcal{T}_h$ :

$$\int_K \sigma_h \cdot \tau \, dx = -\int_K u_h \, \nabla \cdot \tau \, dx + \int_{\partial K} \hat{u}_K \, n_K \cdot \tau \, ds \qquad \forall \tau \in \Sigma(K)$$

$$\int_K \sigma_h \cdot \nabla v \, dx = \int_K f v \, dx + \int_{\partial K} \hat{\sigma}_K \cdot n_K \, v \, ds \qquad \forall v \in P(K)$$

Le choix des flux $\hat{u}_K$ et $\hat{\sigma}_K$ en fonction de $u_h$ et $\sigma_h$ définit entièrement la méthode DG.

### Sommation sur tous les éléments

On somme les formules sur tous les éléments $K \in \mathcal{T}_h$.

$$\int_\Omega \sigma_h \cdot \tau \, dx = -\int_\Omega u_h \, \nabla_h \cdot \tau \, dx + \sum_{K\in\mathcal{T}_h} \int_{\partial K} \hat{u}_K \, n_K \cdot \tau \, ds$$

$$\int_\Omega \sigma_h \cdot \nabla_h v \, dx = \int_\Omega f v \, dx + \sum_{K\in\mathcal{T}_h} \int_{\partial K} \hat{\sigma}_K \cdot n_K \, v \, ds$$

et on remarque que les sommes de bord $\sum_K \int_{\partial K}$ font apparaître chaque interface deux fois. 

### Définitiond des termes de saut et de moyenne

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

### Réécriture des termes de bord avec les opérateurs de saut et de moyenne

On réécrit alors les sommes de bord à l'aide de l'identité suivante :

$$\sum_{K\in\mathcal{T}_h} \int_{\partial K} q_K \, \varphi_K \cdot n_K \, ds = \int_\Gamma [\![q]\!] \cdot \{\varphi\} \, ds + \int_{\Gamma_I} \{q\} \, [\![\varphi]\!] \, ds \tag{1}$$

où $\Gamma$ est l'union de tous les bords d'éléments (intérieurs + bords), $\Gamma_I = \Gamma \setminus \partial\Omega$ est l'ensemble des interfaces intérieures.

### Formulation globale après sommation
 
Après avoir sommé sur tous les éléments et réécrit les termes de bord avec (1), on obtient deux équations globales :
 
$$\int_\Omega \sigma_h \cdot \tau \, dx = -\int_\Omega u_h \nabla_h \cdot \tau \, dx + \int_\Gamma [\![\hat{u}]\!] \cdot \{\tau\} \, ds + \int_{\Gamma_I} \{\hat{u}\}\, [\![\tau]\!] \, ds \tag{2}$$
 
$$\int_\Omega \sigma_h \cdot \nabla_h v \, dx - \int_\Gamma \{\hat{\sigma}\} \cdot [\![v]\!] \, ds - \int_{\Gamma_I} [\![\hat{\sigma}]\!]\{v\} \, ds = \int_\Omega f v \, dx \tag{3}$$
 
L'objectif est maintenant d'éliminer $\sigma_h$ de ce système pour obtenir une équation ne portant que sur $u_h$.
 
### Expression de $\sigma_h$ en fonction de $u_h$
 
Pour transformer le terme $-\int_\Omega u_h \nabla_h \cdot \tau \, dx$ dans (2), on utilise l'intégration par parties globale suivante (qui est une conséquence de l'identité (1)) :
 
$$-\int_\Omega u_h \nabla_h \cdot \tau \, dx = \int_\Omega \nabla_h u_h \cdot \tau \, dx - \int_\Gamma \{\tau\} \cdot [\![u_h]\!] \, ds - \int_{\Gamma_I} [\![\tau]\!]\{u_h\} \, ds$$
 
En substituant dans (2), les termes en $\{\tau\}$ et $[\![\tau]\!]$ se regroupent et font apparaître l'écart entre le flux numérique $\hat{u}$ et la trace de $u_h$ :
 
$$\int_\Omega \sigma_h \cdot \tau \, dx = \int_\Omega \nabla_h u_h \cdot \tau \, dx + \int_\Gamma [\![\hat{u} - u_h]\!] \cdot \{\tau\} \, ds + \int_{\Gamma_I} \{\hat{u} - u_h\}\, [\![\tau]\!] \, ds \tag{4}$$
 
Pour écrire cela de façon compacte, on introduit les **opérateurs de relèvement** $r : [L^2(\Gamma)]^2 \to \Sigma_h$ et $l : L^2(\Gamma_I) \to \Sigma_h$, définis par :
 
$$\int_\Omega r(\varphi) \cdot \tau \, dx = -\int_\Gamma \varphi \cdot \{\tau\} \, ds \qquad \int_\Omega l(q) \cdot \tau \, dx = -\int_{\Gamma_I} q\, [\![\tau]\!] \, ds$$
 
Ces opérateurs relèvent une quantité définie sur les interfaces vers l'espace $\Sigma_h$. Avec cette notation, (4) s'écrit directement comme une expression de $\sigma_h$ en fonction de $u_h$ uniquement :
 
$$\sigma_h = \nabla_h u_h - r([\![\hat{u} - u_h]\!]) - l(\{\hat{u} - u_h\})$$
 
### La forme bilinéaire primale

Il reste à éliminer $\sigma_h$ de (3). Pour cela, on prend $\tau = \nabla_h v$ dans (4), ce qui exprime $\int_\Omega \sigma_h \cdot \nabla_h v \, dx$ en termes de $u_h$ :
 
$$\int_\Omega \sigma_h \cdot \nabla_h v \, dx = \int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx + \int_\Gamma [\![\hat{u} - u_h]\!] \cdot \{\nabla_h v\} \, ds + \int_{\Gamma_I} \{\hat{u} - u_h\}\, [\![\nabla_h v]\!] \, ds$$
 
En substituant dans (3), on obtient une équation portant uniquement sur $u_h$, que l'on écrit sous la forme $B_h(u_h, v) = \int_\Omega f v \, dx$ avec :
 
$$B_h(u_h, v) := \int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx + \int_\Gamma \Big([\![\hat{u} - u_h]\!] \cdot \{\nabla_h v\} - \{\hat{\sigma}\} \cdot [\![v]\!]\Big) \, ds$$
$$+ \int_{\Gamma_I} \Big(\{\hat{u} - u_h\}[\![\nabla_h v]\!] - [\![\hat{\sigma}]\!]\{v\}\Big) \, ds$$
 
C'est la **formulation primale** de la méthode DG. Toute la richesse des différentes méthodes DG réside dans le choix des flux $\hat{u}$ et $\hat{\sigma}$ : c'est ce choix qui détermine la forme concrète de $B_h$ et donc les propriétés de stabilité, consistance et précision de la méthode.

### Le problème discret final
 
On peut maintenant écrire le problème discret sous sa forme définitive. On cherche $u_h \in V_h$ tel que :
 
$$B_h(u_h, v) = \int_\Omega f \, v \, dx \qquad \forall v \in V_h$$
 
C'est une équation variationnelle classique : le membre de droite est le terme source $f$ testé contre $v$, exactement comme dans la formulation faible continue. Le membre de gauche $B_h(u_h, v)$ remplace la forme bilinéaire $\int_\Omega \nabla u \cdot \nabla v \, dx$ du problème continu, en y ajoutant tous les termes d'interface qui assurent la communication entre éléments et la stabilité de la méthode.
 

## Choix du flux numérique : SIPG

SIPG (*Symmetric Interior Penalty Galerkin*) est l'une des méthodes DG les plus utilisées pour les problèmes elliptiques. Elle est obtenue en choisissant les flux numériques de façon à ce que la forme bilinéaire $B_h$ soit **symétrique**, **consistante**, et **stable** moyennant un paramètre de pénalité suffisamment grand.
 
### Choix des flux
 
Le flux scalaire $\hat{u}$ est pris comme la **moyenne** de $u_h$ aux interfaces :
 
$$\hat{u} = \{u_h\} \text{ sur } \Gamma_I, \qquad \hat{u} = 0 \text{ sur } \partial\Omega$$
 
Le flux vectoriel $\hat{\sigma}$ est pris comme la moyenne du gradient brisé, auquel on ajoute un **terme de pénalité** sur le saut de $u_h$ :
 
$$\hat{\sigma} = \{\nabla_h u_h\} - \eta \, [\![u_h]\!] \text{ sur } \Gamma_I, \qquad \hat{\sigma} = \nabla_h u_h - \eta \, u_h \, n \text{ sur } \partial\Omega$$
 
où $\eta > 0$ est le **paramètre de pénalité**, défini sur chaque interface $e$ par $\eta|_e = \eta_0 / h_e$, avec $h_e$ la taille de l'interface et $\eta_0 > 0$ une constante à choisir suffisamment grande.
 
Le terme de pénalité $\eta [\![u_h]\!]$ pénalise le saut de $u_h$ aux interfaces : plus $\eta$ est grand, plus on force $u_h$ à être continue au sens faible à travers les interfaces.
 
### La forme bilinéaire SIPG

En injectant ces choix de flux dans la formulation primale, on obtient :

$$B_h(u_h, v) := \underbrace{\int_\Omega \nabla_h u_h \cdot \nabla_h v \, dx}_{\text{diffusion brisée}} - \underbrace{\int_\Gamma \Big( \{\nabla_h u_h\} \cdot [\![v]\!] + [\![u_h]\!] \cdot \{\nabla_h v\} \Big) \, ds}_{\text{consistance + symétrie}} + \underbrace{\int_\Gamma \eta \, [\![u_h]\!] \cdot [\![v]\!] \, ds}_{\text{pénalité}}$$

Le premier terme est la diffusion brisée, analogue au terme de la formulation continue. Le deuxième assure que la solution exacte satisfait bien la formulation discrète. Le troisième contrôle les sauts aux interfaces et garantit la stabilité.

### Propriétés
 
**Consistance.** La solution exacte $u$ satisfait $B_h(u, v) = \int_\Omega f v \, dx$ pour tout $v \in V_h$, ce qui implique l'orthogonalité de Galerkin :
 
$$B_h(u - u_h, v) = 0 \qquad \forall v \in V_h$$
 
Cela découle du fait que $[\![u]\!] = 0$ et $\{\nabla u\} = \nabla u$ pour la solution exacte, qui est régulière.
 
**Symétrie.** La forme $B_h$ est symétrique : $B_h(u_h, v) = B_h(v, u_h)$. C'est la conséquence directe du signe $-$ devant les deux termes de consistance (un pour $u_h$, un pour $v$). Cette propriété est ce qui distingue SIPG de NIPG (*Non-symmetric* IP), où le signe du second terme est inversé.
 
**Stabilité.** La méthode est stable dès que $\eta_0$ est suffisamment grand, au sens où il existe une constante $C_s > 0$ telle que :
 
$$B_h(v, v) \geq C_s \, \| v \|_{1,h}^2 \qquad \forall v \in V_h$$
 
où $\|v\|_{1,h}^2 = \int_\Omega |\nabla_h v|^2 \, dx + \int_\Gamma \eta \, |[\![v]\!]|^2 \, ds$ est la norme d'énergie naturelle de la méthode.
 
### Paramètre de pénalité $\eta$
 
Le choix $\eta|_e = \eta_0 / h_e$ est standard. En pratique, pour des polynômes de degré $p$ sur des triangles réguliers (ou sur le maillage 1D), une valeur couramment utilisée est $\eta_0 = p(p+1)$, ce qui garantit la stabilité. Une valeur trop petite de $\eta_0$ rend la forme bilinéaire indéfinie ; une valeur trop grande n'affecte pas la consistance mais peut détériorer le conditionnement du système linéaire.
 
### Convergence
 
Sous les hypothèses de régularité $u \in H^{p+1}(\Omega)$, SIPG atteint les taux de convergence optimaux suivants :
 
| Norme | Taux de convergence |
|---|---|
| $\|\cdot\|_{1,h}$ (énergie) | $\mathcal{O}(h^p)$ |
| $\|\cdot\|_{L^2(\Omega)}$ | $\mathcal{O}(h^{p+1})$ |