# Implémentation des EDP - forme faible

## Équations différentielles linéaires

> src/scimba_jax/physical_models/abstract_linear_weak_form.py

Classe abstraite dont héritent tous les EDPs linéaires du schéma numérique. Elle définit l'interface que l'assembleur (`AbstractEllipticDGscheme` ou `AbstractEllipticFEscheme`) attend :
| Méthode / attribut | Rôle |
|---|---|
| A - `fields` | dictionnaire de fonctions non-apprenables (ex: `{"f": lambda x: ...}`) |
| A - `models` | liste des modules apprenables (`eqx.Module`) — e.g. réseaux de neurones |
| M - `bilinear_form(u, v)` | à implémenter — retourne l'intégrande $B(u,v)(x)$ comme `ParamFunc` |
| M - `linear_form(v)` | à implémenter — retourne l'intégrande $L(v)(x)$ comme `ParamFunc` |
| M - `list_learnable_fields()` | retourne les champs apprenables wrappés en `ParamFunc` |

### Problème elliptique linéaire

> src/scimba_jax/physical_models/classical_weakform/diffusion_advection_reaction_weak_form.py

Pour le problème $-\nabla \cdot (a \nabla u) + b \cdot \nabla u + c u = f$ sur $\Omega$, la formulation faible donne :
$$\int_\Omega a \nabla u \cdot \nabla v + b \cdot \nabla u\, v + c u v = \int_\Omega f\, v \quad \forall v$$
`DiffusionAdvectionReactionWeakForm` hérite de `AbstractLinearWeakForm` et implémente les deux formes comme des opérations sur les `ParamFunc` (voir `param_func.md`).


### Le Laplacien 

> src/scimba_jax/physical_models/classical_weakform/laplacian_weak_form.py
<!-- > src/scimba_jax/physical_models/classical_weakform/laplacian_weak_form_learnable_source.py -->

Pour le problème $-\Delta u = f$ sur $\Omega$, la formulation faible donne :
$$\int_\Omega \nabla u \cdot \nabla v = \int_\Omega f\, v \quad \forall v$$
`LaplacianWeakForm` hérite de `AbstractLinearWeakForm` et implémente les deux formes comme des opérations sur les `ParamFunc`, par exemple :
```python
def bilinear_form(u, v):
    return u.gradient_x().dot(v.gradient_x())  # ∇u · ∇v  (intégrande)
def linear_form(v):
    return self.fields["f"] * v                # fv  (intégrande)
```

## Équations différentielles non-linéaires

<span style="color: red;">(Non implémenté)</span>