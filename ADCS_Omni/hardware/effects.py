def apply_effects(value, effects, dt, rng=None):
    for effect in effects:
        value = effect.apply(value, dt, rng)
    return value
