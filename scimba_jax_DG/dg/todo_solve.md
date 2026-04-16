# <span style="color:blue"><b>TODO :</b></span> Classe `EllipticDGscheme` - Résolution

## Premiers tests

- [x] Modifier `__call__`des flux pour retourner un seul élément (ancienne version à reprendre)
- [x] Implémenter le flux SIPG  et tester l'assemblage du flux sur un problème simple (dofs à 1)
- [x] Ajouter les notes sur le flux SIPG en présentant l'exemple.
- [x] Ajouter les conditions de Dirichlet et tester à nouveau l'assemblage du flux
- [x] Modifier le calcul de l'erreur $L^2$ pour qu'elle soit calculée maille par maille (et non pas globalement)
- [x] Tester le problème de Poisson avec solution polynomiale d'ordre 2 (avec $\Rightarrow f=1$) et comparer la solution numérique à la solution exacte $\Rightarrow$ doit être exact pour $k=2$
- [x] Tester Poisson avec $f$ sinusoïdal et comparer la solution numérique à la solution exacte en vérifiant les convergences en $L^2$ et $L^\infty$ - `solve_laplacian.py`
- [x] Tester sur un problème élliptique plus complexe (ajouter une matrice de diffusion) - `solve_diffusion.py`

## Quelques extensions

- [x] **WARNING !!!** Réparer le flux centré.
- [x] Renvoyer deux flux indépendants et enlever les zeros dans le `EllipticDGscheme`.
- [x] Rajouter un flux de bord... dans le `assembly_local_flux_terme_pure` qui ne s'occupe que des interfaces intérieures, et un flux spécial pour les bords.
- [x] Tester 2 autres flux en plus de SIPG (regarder ref).
- [x] Tester avec de l'advection ($b$ non nul).
- [x] Ajouter de la réaction ($c$ non nul).
- [ ] Tester avec des termes sources différents
- [x] Regarder les perfs : pourquoi ça mets autant de temps à tourner ?

## Pour aller plus loin

- [x] En plus du Newton, ajouter une version matrix free (pour le linéaire : coder un cg matrix-free à la main qui prend en entrée une fonction qui est le produit matrice vecteur VJP, écrire le newton matrix free). Tester qu'on a les mêmes résultats que la version avec matrice.
- [x] Mettre solve comme une fonction statique de la classe `EllipticDGscheme` qui appelle le newton (fct en dehors de la classe qui prend un résidual en entrée : soit version standard soit matrix free).
- [x] Créer une nouvelle classe `dg_approximation_space` qui est pas projection mais elliptic. Dans le `get_intermediare_value`, on résout le problème pour calculer les dofs en faisant le `EllipticDGscheme.solve`.
- [x] Si tout marche bien, on est censé pouvoir faire l'optimisation du Laplacien. On résout le lap en dg avec très peu de mailles, ensuite on utilise un PINN qui prend en entrée le `dg_elliptic` qui va optimiser les bases et le mapping de façon à résoudre au mieux le laplacien.
- [ ] Plus tard : tester si on apprend un flux, qu'est-ce qui se passe...
- [ ] Une fois que tout sera ça bon, on pourra faire le FEM... le FEM y a pas de terme de flux, c'est juste le terme de volume où il faut modifier un truc dedans.

## Suite

- [x] Enlever diff et diff/ad (garder que le cas général A,b,c)
- [ ] Tester temps exécution entre newton normal et mat free pour la convergence du laplacien
		-> tester de jiter "matvec" dans version matrix-free
- [ ] Pour DG + PINN :
	- [x]  Tester patchwise + mapping
	      [(voir)](images/dgelliptic_patchwise_basis_with_mapping.png)
	- [x] Tester version bases cellwise
	      [(voir)](images/dgelliptic_cellwise_basis.png)
	- [ ] Tester avec version matrix-free (Attention : point suivant)
- [ ] Dans les 2 cas du newton, il faut faire le VJP où on a besoin de faire une inversion de matrice (pour le matrix-free on fait un cg où matvec c'est J^T * v avec une iter pour le cg mais pas de newton)
- [ ] Tester le non-linéaire avec
      $$-\nabla\cdot(A(u,x)\nabla u) + b(u,x)\cdot\nabla u + c(u,x) = f(x)$$
      avec $u=u_\theta(x)$.
      Attention : la composition peut pas prendre u (écrire la lambda fct lambda x: x^2 où x est u) 
- [ ] Vérifier nb_dofsl peut-être pas bon dans le cadre bases field ou vec 