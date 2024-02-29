"""Loss functions for DeepMoS."""

import equinox as eqx
from datasetv2 import AudioDataset
from einops import repeat
from jax import numpy as jnp
from jax.random import split
from jaxtyping import Array, Float, PRNGKeyArray

from mos.models import Model


def _frame_and_global_loss_fn(data: AudioDataset, pred: Float[Array, " batch"]) -> Float[Array, "batch time"]:
    """Compute the loss of a batch for each frame and the mean of the batch."""
    mos = repeat(data.mos, "batch -> batch time 1", time=pred.shape[1])
    frame_loss = jnp.square(pred - mos).mean()
    mean_loss = jnp.square(pred.mean(axis=1) - mos.mean(axis=1)).mean()
    return frame_loss + mean_loss


def multi_head_batch_loss(
    model: Model,
    data: AudioDataset,
    model_state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, "batch time"], tuple[eqx.nn.State, Float[Array, " batch"]]]:
    """Compute the loss of a batch.

    Args:
        model: Model to use.
        data: Data of the batch.
        model_state: State of the model.
        key: Key to use for randomness.

    Returns:
        Loss of the batch.
    """
    pred, model_state = eqx.filter_vmap(model, in_axes=(0, 0, None, 0), out_axes=(0, None), axis_name="batch")(
        data.ref, data.deg, model_state, split(key, len(data.mos))
    )

    total_loss = _frame_and_global_loss_fn(data, pred)
    return total_loss, (model_state, pred)
