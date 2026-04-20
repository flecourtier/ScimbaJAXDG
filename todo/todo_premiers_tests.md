# ToDo - Premiers tests

## Idée générale

On écrit $u_\theta$ sous la forme :

$$u_\theta(x) = \sum_{i=1}^N \alpha_i \varphi_i(x,\beta_i)$$

avec $\theta = (\alpha, \beta)$ et où chaque $\varphi_i$ est un réseau de neurones avec des paramètres $\beta_i$.

Pour ne pas avoir à optimiser les $\alpha_i$ et les $\beta_i$ simultanément ou de manière alternée, on peut utiliser la structure linéaire du problème pour résoudre exactement les $\alpha_i$ en fonction des $\beta_i$. Plus précisément, pour un ensemble donné de paramètres $\beta$, on peut construire la matrice $A(\beta)$ et le vecteur $b$ tels que :

$$A(\beta) \vec\alpha = b$$

Ainsi l'inconnu $u_\theta$ devient :
$$u_\beta(x) = \sum_{i=1}^N \left(A(\beta)^{-1} b\right)_i \varphi_i(x,\beta_i)$$

et le problème de minimisation devient un problème d'optimisation uniquement sur les paramètres $\beta$ :
$$\beta^* = \arg\min_\beta \|\Delta u_\beta + f\|_{L^2}.$$

## Devoirs

### Premier point

**Objectif :** Implémenter la résolution de problèmes linéaires et non linéaires en 1D à l'aide de schémas de différences finies en utilisant JAX pour les calculs différentiables.

On considére le problème suivant en 1D:

$$\left\{\begin{align*}
-u''(x)+g(u) &= f(x),\quad x\in(0,1) \\
u(0) = u(1) &= 0\end{align*}\right.$$

pour $g=\alpha u$ (linéaire) et pour $g=\alpha u^2$ (non linéaire).

On applique un schéma différences finies centré d'ordre 2 sur un maillage uniforme : 

$$\left\{\begin{align*}
-u_{i+1}+2u_i-u_{i-1} + \alpha h^2 u_i &= h^2f_i,\quad i=1,\ldots,N-1 \\
u_0 = u_N &= 0\end{align*}\right.$$ 

- Dans le cas où le système est **linéaire**, on peut donc le résoudre **directement** :

    $$A\mathbf{u} = \mathbf{f}$$

    avec $A$, $\mathbf{u}$ et $\mathbf{f}$ définis par :

    $$A = \frac{1}{h^2}\begin{pmatrix}
    2+\alpha h^2 & -1 & 0 & \cdots & 0 \\
    -1 & 2+\alpha h^2 & -1 & \cdots & 0 \\
    0 & -1 & 2+\alpha h^2 & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \cdots & 2+\alpha h^2
    \end{pmatrix}, \quad \mathbf{u} = \begin{pmatrix}
    u_1 \\ u_2 \\ u_3 \\ \vdots \\ u_{N-1}
    \end{pmatrix}, \quad \mathbf{f} = \begin{pmatrix}
    f_1 \\ f_2 \\ f_3 \\ \vdots \\ f_{N-1}
    \end{pmatrix}$$

- Dans le cas où le système est **non-linéaire**, on utilise donc la **méthode de Newton-Raphson** pour le résoudre :

    $$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} - J_F(\mathbf{u}^{(k)})^{-1} F(\mathbf{u}^{(k)})$$

    où $F(\mathbf{u})$ est le résidu et $J_F$ sa Jacobienne.

### Second point 

**Objectif :** Etudier la dépendance de la solution par rapport à $\alpha$ en utilisant la différentiation automatique de JAX pour calculer les dérivées de la solution par rapport à $\alpha$. Appliquer le théorème des fonctions implicites pour comprendre comment la solution change lorsque $\alpha$ varie, et vérifier les résultats obtenus par différentiation automatique avec des calculs analytiques ou numériques.

### Troisième point

**Objectif :** Même objectif mais avec du Matrix-Free. 

- *V1 :* Tester avec la librairie `jaxopt` ([Nonlinear least squares](https://jaxopt.github.io/stable/nonlinear_least_squares.html)). 

- *V2 :* Implémenter la méthode de Newton à la main.

    - Forward (Méthode de Newton)

        **Initialisation :** $u_0$

        **Pour $i = 1, \dots$ :**

        $$ J(u_k) \delta u = -F(u_k) $$

        $$ u_{k+1} = \dots $$

        > **Note :** On résout ce système avec un **Gradient conjugué** (À écrire) qui prend l'opérateur **JVP** de $F(u)$. $\longrightarrow$ *par rapport à $u$*

    - Backward (Méthode de l'Adjoint)

        Prend en entrée $u^*$ (la solution convergée) et une direction $v$.

        $$ J(u^*)^T r = v $$

        > **Note :** Résolu via **Gradient conjugué** en utilisant le **VJP** de $F(u^*)$. $\longrightarrow$ *par rapport à $u$*

    - Sensibilité / Gradient

        $$ \frac{\partial u^*}{\partial \alpha} = \left( \frac{\partial F}{\partial \alpha} \right)^T r $$

        > **Note :** Correspond au **VJP de $F$**  $\longrightarrow$ *par rapport à $\alpha$*