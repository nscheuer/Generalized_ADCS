from dataclasses import dataclass

@dataclass
class DisturbanceMode:
    add_bias: bool               # Should the bias be added to dynamics?
    add_noise: bool              # Should noise be added to dynamics?
    update_bias: bool            # Should bias evolve (random walk)?
    update_noise: bool           # Should noise be resampled each call?