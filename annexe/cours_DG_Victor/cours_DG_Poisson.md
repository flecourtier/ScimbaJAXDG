# Cours DG - Problème de Poisson

(écrit par Claude AI, non relu)


---

## 1. Équation de Poisson

$$-\Delta u = f \qquad u : \Omega \rightarrow \mathbb{R}, \quad \Omega \subset \mathbb{R}^d$$

avec des conditions aux limites de Dirichlet homogènes :

$$u = 0 \quad \text{sur } \partial\Omega$$

**Exemple :** $d=1$, $\Omega = (0,1)$, $f(x) = \pi^2 \sin(\pi x)$, solution exacte $u(x) = \sin(\pi x)$.

---

## 2. Espaces fonctionnels et Maillage

$u \in V$ (espace fonctionnel de dim infinie)

**Exemple :** $V = H^1_0(\Omega)$

$V_{h,i} \subset L^2(\Omega)$ (espace DG par morceaux de dim finie : $p+1$) $\leftarrow$ dépend du maillage et de la cellule $i$

$$ V_{h,i} = \text{Span}(\varphi_i^0, \dots, \varphi_i^p) $$

où $i$ est le numéro de cellule du maillage $\mathcal{M}$.

> **Note :** Contrairement à $H^1_0$, les fonctions DG sont **discontinues aux interfaces** — on travaille dans $V_h \subset L^2(\Omega)$, pas dans $H^1(\Omega)$.

### Maillage 1D

$$i \in \mathcal{M} \subset \mathbb{Z}$$
$$x_i = a + ih + \frac{h}{2}$$
$$x_{i+\frac{1}{2}} = a + \left(i+\frac{1}{2}\right)h + \frac{h}{2}$$

---

## 3. Bases polynomiales locales

$\forall i \in \mathcal{M}, \forall x \in \left(x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}\right) = C_i$

**Exemple :** Base de Taylor
$$\varphi_i^{(k)}(x) = \frac{(x-x_i)^k}{k!} \qquad \forall k \in [\![0, p]\!] \qquad (\circledast)$$

---

## 4. Développement de la solution

$(\circledast) \quad \forall i \in \mathcal{M}, \forall x \in C_i, \quad u_{h,i}(x) = \sum_{j=0}^p u_{i,j}\, \varphi_i^{(j)}(x)$

**Exemple :** $p=1$
$$u_{h,i}(x) = u_{i,0} + u_{i,1}(x-x_i)$$

$\longrightarrow$ ($p+1$) degrés de liberté (ddl) par cellule.

---

## 5. Solution globale et Projection

$$\forall x \in \Omega \qquad u_h(x) = \sum_{i \in \mathcal{M}} u_{h,i}(x)\,\mathbb{1}_{\{x \in C_i\}}$$

Soit $i \in \mathcal{M}$, on cherche $u_{h,i}$ telle que le résidu $E(u_{h,i}) = -\Delta u_{h,i} - f$ soit orthogonal à $V_{h,i}$ :

$$\langle E(u_{h,i})\,,\, \varphi_i^{(k)} \rangle_{L^2} = 0 \qquad \forall k \in [\![0, p]\!]$$

---

## 6. Formulation variationnelle locale

$$\forall i \in \mathcal{M},\quad \forall k \in [\![0, p]\!]$$

$$-\int_{C_i} \Delta u_{h,i}(x)\, \varphi_i^{(k)}(x)\, dx = \int_{C_i} f(x)\, \varphi_i^{(k)}(x)\, dx \qquad (\circledast)$$

On remplace par le développement de $u_{h,i}$ :

$$u_{h,i}(x) = \sum_{j=0}^p u_{i,j}\, \varphi_i^{(j)}(x)$$

Le terme de gauche devient :

$$-\int_{C_i} \Delta\!\left(\sum_{j=0}^p u_{i,j}\, \varphi_i^{(j)}(x)\right) \varphi_i^{(k)}(x)\, dx = -\sum_{j=0}^p u_{i,j} \int_{C_i} \frac{d^2}{dx^2}\varphi_i^{(j)}(x)\, \varphi_i^{(k)}(x)\, dx$$

---

## 7. Intégration par parties et Flux aux interfaces

On intègre par parties le terme de gauche de $(\circledast)$ :

$$-\int_{C_i} \Delta u_{h,i}\, \varphi_i^{(k)}\, dx = \underbrace{\int_{C_i} \nabla u_{h,i}(x) \cdot \nabla \varphi_i^{(k)}(x)\, dx}_{\textcolor{red}{\text{Terme \textcircled{1}}}} \underbrace{- \left[\frac{d}{dx} u_{h,i}(x)\, \varphi_i^{(k)}(x)\right]_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}}}_{\textcolor{red}{\text{Terme \textcircled{2}}}}$$

On développe le terme de bord :

$$\textcircled{2} = -\frac{d}{dx}u_{h,i}(x_{i+\frac{1}{2}})\,\varphi_i^{(k)}(x_{i+\frac{1}{2}}) + \frac{d}{dx}u_{h,i}(x_{i-\frac{1}{2}})\,\varphi_i^{(k)}(x_{i-\frac{1}{2}})$$

aux interfaces, $\nabla u_{h,i}$ est **discontinu** : il faut introduire un flux numérique.

---

## 8. Flux numérique elliptique

Aux interfaces, la solution et son gradient sont discontinus. On introduit un **flux numérique** $\mathcal{F}$ qui approche $\nabla u \cdot n$ de manière consistante, stable et conservative.

$$\frac{d}{dx}u_{h,i}(x_{i+\frac{1}{2}}) \simeq \mathcal{F}\!\left(u_{h,i}(x_{i+\frac{1}{2}}),\, u_{h,i+1}(x_{i+\frac{1}{2}}),\, \nabla u_{h,i}(x_{i+\frac{1}{2}}),\, \nabla u_{h,i+1}(x_{i+\frac{1}{2}})\right)$$

**Exemple : Flux SIPG (Symmetric Interior Penalty Galerkin)**

$$\mathcal{F}(u_L, u_R, \nabla u_L, \nabla u_R, v_L, v_R, \nabla v_L, \nabla v_R, n) =$$
$$-\{\!\{ \nabla u \}\!\} \cdot n\, [\![v]\!] - \{\!\{ \nabla v \}\!\} \cdot n\, [\![u]\!] + \frac{\eta}{h}\, [\![u]\!]\, [\![v]\!]$$

où les **moyennes** et **sauts** aux interfaces sont définis par :

$$\{\!\{ w \}\!\} = \frac{w_L + w_R}{2}, \qquad [\![w]\!] = w_R - w_L$$

et $\eta > 0$ est un paramètre de pénalisation (suffisamment grand pour la stabilité).

> **Note :** Le terme $-\{\!\{\nabla v\}\!\} \cdot n\, [\![u]\!]$ assure la symétrie de la forme bilinéaire. Le terme $\frac{\eta}{h}[\![u]\!][\![v]\!]$ pénalise le saut de la solution aux interfaces pour imposer faiblement la continuité.

---

## 9. Schéma numérique final

En assemblant tous les termes, l'équation $(\circledast)$ donne pour chaque cellule $C_i$ et chaque $k \in [\![0,p]\!]$ :

$$\underbrace{\sum_{j=0}^p u_{i,j} \int_{C_i} \frac{d}{dx}\varphi_i^{(j)}(x)\, \frac{d}{dx}\varphi_i^{(k)}(x)\, dx}_{S_i(U_i)\;\textcolor{red}{\textcircled{1}}}$$

$$-\; \underbrace{\mathcal{F}\!\left(\dots, x_{i+\frac{1}{2}}\right) \varphi_i^{(k)}(x_{i+\frac{1}{2}}) - \mathcal{F}\!\left(\dots, x_{i-\frac{1}{2}}\right) \varphi_i^{(k)}(x_{i-\frac{1}{2}})}_{F_i(U_{i-1}, U_i, U_{i+1})\;\textcolor{red}{\textcircled{2}}}$$

$$= \underbrace{\int_{C_i} f(x)\, \varphi_i^{(k)}(x)\, dx}_{b_i\;\textcolor{red}{\textcircled{3}}}$$

ce qui donne un **système linéaire** $A\, U = b$ (pas d'évolution temporelle ici).

> **Note :** Contrairement au transport, le problème de Poisson est **stationnaire** — on résout un système linéaire global, pas une intégration en temps.

---

## 10. Conditions aux limites

Les CL de Dirichlet $u = 0$ sur $\partial\Omega$ sont imposées **faiblement** via le flux numérique sur les faces de bord $\partial C_i \cap \partial\Omega$ :

On pose $u_{\text{ext}} = 0$ (valeur ghost) et on applique le flux SIPG de la même façon qu'aux interfaces intérieures :

$$\mathcal{F}_{\partial}\!\left(u_L, u_{\text{ext}}, \nabla u_L, \nabla u_{\text{ext}}\right) = -\nabla u_L \cdot n - \nabla v_L \cdot n\,(u_L - u_{\text{ext}}) + \frac{\eta}{h}\,(u_L - u_{\text{ext}})\, v_L$$

> **Note :** L'imposition faible des CL est une caractéristique de la méthode DG : on ne force pas $u_h = 0$ sur le bord mais on pénalise l'écart via le flux.
