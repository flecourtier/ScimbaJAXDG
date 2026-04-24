# <span style="color:blue"><b>TODO :</b></span> Classe `EllipticDGscheme` (solve DG)

## Solve DG - 1D

## Sans apprentissage

- [x] Modifier `__call__`des flux pour retourner un seul élément (ancienne version à reprendre)
- [x] Implémenter le flux SIPG  et tester l'assemblage du flux sur un problème simple (dofs à 1)
- [x] Ajouter les notes sur le flux SIPG en présentant l'exemple.
- [x] Ajouter les conditions de Dirichlet et tester à nouveau l'assemblage du flux
- [x] Modifier le calcul de l'erreur $L^2$ pour qu'elle soit calculée maille par maille (et non pas globalement)
- [x] Tester le problème de Poisson avec solution polynomiale d'ordre 2 (avec $\Rightarrow f=1$) et comparer la solution numérique à la solution exacte $\Rightarrow$ doit être exact pour $k=2$
- [x] Tester Poisson avec $f$ sinusoïdal et comparer la solution numérique à la solution exacte en vérifiant les convergences en $L^2$ et $L^\infty$ - `solve_laplacian.py`
- [x] Tester sur un problème élliptique plus complexe (ajouter une matrice de diffusion) - `solve_diffusion.py`
- [x] **WARNING !!!** Réparer le flux centré.
- [x] Renvoyer deux flux indépendants et enlever les zeros dans le `EllipticDGscheme`.
- [x] Rajouter un flux de bord... dans le `assembly_local_flux_terme_pure` qui ne s'occupe que des interfaces intérieures, et un flux spécial pour les bords.
- [x] Tester 2 autres flux en plus de SIPG (regarder ref).
- [x] Tester avec de l'advection ($b$ non nul).
- [x] Ajouter de la réaction ($c$ non nul).
- [x]  Enlever diffusion et diffusion/advection (garder que le cas général avec A,b,c)
- [ ] Tester avec des termes sources différents
- [ ] <span style="color:red"><b>(Point 3)</b></span> Faire comme pour les résidus où une EDP c'est pas juste une weak form mais elle contient une weak form intérieur et une de bord.
- [ ] <span style="color:red"><b>(Point 5)</b></span>
	Validation de system (avec  2 laplacien couplés ).

**Performances:**

- [x] Regarder les perfs du solve_laplacian.
- [x] En plus du Newton, ajouter une version matrix free (pour le linéaire : coder un cg matrix-free à la main qui prend en entrée une fonction qui est le produit matrice vecteur VJP, écrire le newton matrix free). Tester qu'on a les mêmes résultats que la version avec matrice.

## Avec apprentissage

- [x] Mettre solve comme une fonction statique de la classe `EllipticDGscheme` qui appelle le newton (fct en dehors de la classe qui prend un résidual en entrée : soit version standard soit matrix free).
- [x] Créer une nouvelle classe `dg_approximation_space` qui est pas projection mais elliptic. Dans le `get_intermediare_value`, on résout le problème pour calculer les dofs en faisant le `EllipticDGscheme.solve`.
- [x] Si tout marche bien, on est censé pouvoir faire l'optimisation du Laplacien. On résout le lap en dg avec très peu de mailles, ensuite on utilise un PINN qui prend en entrée le `dg_elliptic` qui va optimiser les bases (patchwise) de façon à résoudre au mieux le laplacien.
- [x] Tester de rajouter un mapping entraînable en plus des bases patchwise.
- [x] Tester version bases cellwise
- [ ] Tester bases patchwise avec version matrix-free. Dans les 2 cas, il faut faire le VJP avec une inversion de matrice (pour le matrix-free on fait un cg où matvec c'est $J^T * v$ avec une iter pour le cg mais pas de newton)
		-> tester de jiter "matvec" dans version matrix-free
- [ ] <span style="color:red"><b>(Point 1)</b></span>
	Tester le non-linéaire avec $-\nabla\cdot(A(u,x)\nabla u) + b(u,x)\cdot\nabla u + c(u,x) = f(x)$ où $u=u_\theta(x)$. 
	Attention : la composition ne peut pas prendre $u$ directement (écrire la lambda fonction comme lambda x: x^2 où x est u). 
- [ ] Plus tard : tester si on apprend un flux, qu'est-ce qui se passe...

**Performances:**

- [ ] Regarder les modifs d'Emmanuel et de Victor.
- [ ] Tester temps exécution entre newton normal et mat free pour la convergence du laplacien
- [ ] Tester de demander à Claude de faire un code dg tout simple pour résoudre laplacien P1 sur un carré et comparer temps d'execution avec version scimba_jax_dg
- [ ] (**Michel**) Est-ce qu'avec une base enrichie on gagne en ordre ? ou en constante de convergence ?  
## Solve DG - Multidimensionnel

- [x] Adapter `Mesh` et `EllipticDGscheme` pour le cas multidimensionnel.
- [x] Tester un problème de Poisson en $\mathbb{P1}$ sur un carré et vérifier les convergences.
- [ ] <span style="color:red"><b>(Point 6)</b></span>
	Tester en 3D.

## Solve FEM - 1D

- [x] Modifier Variables en VariablesDG.
- [ ] <span style="color:red"><b>(Point 2)</b></span> Implémenter VariablesFE (en plus de VariablesDG). On a moins de dofsl car il y a des dofs partagés. Ajouter a une fonction qui permet de récupérer à partir de la cellule local, l'indice global du dofs dans dofsl.
- [ ] Tester un laplacien Dirichlet avec VariablesFE.
- [x] <span style="color:red"><b>(Point 4)</b></span> Implémenter les bases de Lagrange + tester.
- [ ] Adapter l'implémentation des bases de Taylor [(Thèse - Taylor DG)](https://theses.hal.science/tel-00765575/document).
- [ ] Tester d'implémenter $\varphi$-FEM Dirichlet.