# Implémentation des EDP

ANCIENNE VERSION !!!!!

## 8. Équations différentielles - linéaire, forme faible (`AbstractLinearWeakForm`)

> src/scimba_jax/physical_models/abstract_linear_weak_form.py

Classe abstraite dont héritent tous les PDEs linéaires du schéma DG. Elle définit l'interface que l'assembleur `AbstractEllipticDGscheme` attend :
| Méthode / attribut | Rôle |
|---|---|
| A - `fields` | dictionnaire de fonctions non-apprenables (ex: `{"f": lambda x: ...}`) |
| A - `models` | liste des modules apprenables (`eqx.Module`) — e.g. réseaux de neurones |
| M - `bilinear_form(u, v)` | à implémenter — retourne l'intégrande $B(u,v)(x)$ comme `ParamFunc` |
| M - `linear_form(v)` | à implémenter — retourne l'intégrande $L(v)(x)$ comme `ParamFunc` |
| M - `list_learnable_fields()` | retourne les champs apprenables wrappés en `ParamFunc` |

### Le Laplacien (`LaplacianWeakForm`)  

> src/scimba_jax/physical_models/classical_weakform/laplacian_weak_form.py

Pour le problème $-\Delta u = f$ sur $\Omega$, la formulation faible donne :
$$\int_\Omega \nabla u \cdot \nabla v = \int_\Omega f\, v \quad \forall v$$
`LaplacianWeakForm` hérite de `AbstractLinearWeakForm` et implémente les deux formes comme des opérations sur les `ParamFunc` (voir section 7) :
```python
def bilinear_form(u, v):
    return u.gradient_x().dot(v.gradient_x())  # ∇u · ∇v  (intégrande)
def linear_form(v):
    return self.fields["f"] * v                # fv  (intégrande)
```