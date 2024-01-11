"""Loss functions for DeepMoS."""
import equinox as eqx
from dataset import AudioDataset, AudioDatasetRef
from jax import numpy as jnp
from jax.random import split
from jaxtyping import Array, Float, PRNGKeyArray
from models import DeepMos, MultiEncoderMos
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
    )(data.deg, model_state, split(key, len(data.deg)))
    nll = -tfp.distributions.Normal(mean, variance).log_prob(data.score).mean()

    return nll, model_state


def batch_loss_mse(
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
    (mean, _), model_state = eqx.filter_vmap(model, in_axes=(0, None, 0), out_axes=(0, None), axis_name="batch")(
        data.deg, model_state, split(key, len(data.deg))
    )
    mse = jnp.square(mean - data.score).mean()

    return mse, model_state


def multi_head_batch_loss(
    model: MultiEncoderMos,
    data: AudioDatasetRef,
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
    mean, model_state = eqx.filter_vmap(model, in_axes=(0, 0, None, 0), out_axes=(0, None), axis_name="batch")(
        data.ref, data.deg, model_state, split(key, len(data.mos))
    )
    frame_loss = jnp.square(mean - data.mos).mean()
    mean_loss = jnp.square(mean.mean(axis=1) - data.mos).mean()

    total_loss = frame_loss + mean_loss
    return total_loss, model_state
