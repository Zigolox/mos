"""Evaluation functions for DeepMOS."""
import equinox as eqx
from dataset import AudioDataset
from jax import numpy as jnp
from jax.random import split
from jaxtyping import Array, Float, PRNGKeyArray
from models import DeepMos
from typing import Callable
from scipy.stats import spearmanr


def evaluate(
    model: DeepMos,
    model_state: eqx.nn.State,
    data: AudioDataset,
    loss_fn: Callable[[DeepMos, AudioDataset, eqx.nn.State, PRNGKeyArray], Float[Array, "data_size"]],
    key: PRNGKeyArray,
) -> tuple[float, float, float]:
    """Evaluate a model.

    Args:
        model: Model to evaluate.
        model_state: State of the model.
        data: Data to use.
        key: Key to use for randomness.
        loss_fn; Loss function to use.
        key: Key to use for randomness.

    Returns:
        Loss of the model.
        Spearmann correlation of the model.
        Pearson correlation of the model.
    """
    # Put the model in inference mode.
    eqx.nn.inference_mode(model)
    # Compute the loss in regards to the model parameters.
    loss, _ = loss_fn(model, data, model_state, key)
    # Compute the model predictions.
    # (mean, _), _ = eqx.filter_vmap(model, in_axes=(0, None, 0), out_axes=(0, None), axis_name="batch")(
    #    data.deg, model_state, split(key, len(data.deg))
    # )
    # pred = jnp.ravel(mean.mean(axis=1))
    # print(pred.shape, data.score.shape)
    # Spearmann correlation
    # spearmann = spearmanr(data.score, pred)[0]
    # Pearson correlation
    # pearson = jnp.corrcoef(data.score, pred)[0, 1]
    eqx.nn.inference_mode(model, False)

    return loss, 0, 0  # spearmann, pearson
