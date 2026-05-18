# Résultats numériques (à conserver pour la thèse)

## DG

- [ ] Mettre au propre tous les résultats obtenus en DG jusque là
- [ ] <span style="color:green"><b>(Point 3)</b></span> Courbes temps/erreur pour différentes tailles de maillage (laplacien 1D et 2D) : 
      Fixer la taille du maillage, ensuite je regarde temps et erreur obtenu en dg standard, dg patchwise, dg cellwise (inclure temps d'entraînement mais pas la compilation)
- [ ] <span style="color:green"><b>(Point 6)</b></span> Faire une étude d'hyperpramètre (en 1D, 2D et 3D) :
      Comment l'erreur et le temps de calcul se comportent quand on fait varier : la taille des réseau, le nombre de base, le nombre de maille etc.
## FEM

- [ ] Tester si on obtient les mêmes résultats que dans le papier précédent (base patchwise) :
	-  On pré-train un PINN, on l'utilise dans la base patchwise et on vérifie qu'on a pareil que Enriched FEM
	- On compare les résultats en continuant l'apprentissage avec le réseau dans le FEM

## ROMs
