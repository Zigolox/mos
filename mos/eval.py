"""Evaluation functions for DeepMOS."""

from typing import Callable

import equinox as eqx
import grain.python as grain
from jax import lax, numpy as jnp
from jaxtyping import PRNGKeyArray
from scipy.stats import spearmanr

from mos.models import Model


def evaluate(
    model: Model,
    model_state: eqx.nn.State,
    data_loader: grain.DataLoader,
    loss_fn: Callable,
    key: PRNGKeyArray,
) -> tuple[float, float, float]:
    """Evaluate a model.

    Args:
        model: Model to evaluate.
        model_state: State of the model.
        data_loader: Validation dataloader
        loss_fn: Loss function to use.
        key: Key to use for randomness.

    Returns:
        Loss of the model.
        Spearmann correlation of the model.
        Pearson correlation of the model.
    """
    # Put the model in inference mode.
    eqx.nn.inference_mode(model, True)

    # Compute the loss in regards to the model parameters.
    total_loss, total_pred, mos = [], [], []
    for data in data_loader:
        loss, (_, pred) = lax.map(lambda data: loss_fn(model, data, model_state, key), data)
        total_loss += list(loss.ravel())
        total_pred += list(pred.mean(axis=2).ravel())
        mos += list(data.mos.ravel())

    # Compute the mean loss.
    loss = jnp.array(total_loss).mean()
    # Spearmann correlation
    spearmann = spearmanr(mos, total_pred)[0]
    # Pearson correlation
    pearson = jnp.corrcoef(jnp.array(mos), jnp.array(total_pred))[0, 1]
    # Put the model back in training mode.
    eqx.nn.inference_mode(model, False)
    return float(loss), float(spearmann), float(pearson)
