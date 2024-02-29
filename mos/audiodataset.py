"""Dataset for audio files."""

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array


class AudioDataset(eqx.Module):
    """Dataset for audio files."""

    ref: Array = eqx.field(converter=jnp.asarray)
    deg: Array = eqx.field(converter=jnp.asarray)
    mos: Array = eqx.field(converter=jnp.asarray)
