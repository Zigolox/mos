"""Loss functions for DeepMoS."""
import equinox as eqx
from dataset import AudioDataset
from jax.random import split
from jaxtyping import Array, Float, PRNGKeyArray
from models import DeepMos
from tensorflow_probability.substrates import jax as tfp


def batch_loss(
    model: DeepMos,
    data: AudioDataset,
    model_state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, "batch time"], eqx.nn.State]:
    """Compute the loss of a batch.

    Args:
        model: Model to use.
        data: Data of the batch.
        model_state: State of the model.
        key: Key to use for randomness.

    Returns:
        Loss of the batch.
    """
    (mean, variance), model_state = eqx.filter_vmap(
        model, in_axes=(0, None, 0), out_axes=(0, None), axis_name="batch"
    )(data.wav, model_state, split(key, len(data.wav)))
    nll = -tfp.distributions.Normal(mean, variance).log_prob(data.score).mean()

    return nll, model_state
