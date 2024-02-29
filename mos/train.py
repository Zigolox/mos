"""Module responsible for the training loop."""

from functools import partial
from typing import Callable

import equinox as eqx
import grain.python as grain
from jax import lax
from jax.random import split
from jaxtyping import PRNGKeyArray
from optax import GradientTransformation, OptState

import mos.log as log
from mos.datasetv2 import AudioDataset
from mos.models import Model
from mos.utils import infinite_rng_split


def step(
    model: Model,
    data: AudioDataset,
    opt_state: OptState,
    optim: GradientTransformation,
    model_state: eqx.nn.State,
    loss_fn: Callable,
    key: PRNGKeyArray,
):
    """Perform a single training step.

    Args:
        model: Model to train.
        data: Data to use.
        opt_state: State of the optimizer.
        optim: Optimizer to use.
        model_state: Batch norm state.
        loss_fn: Loss function to use.
        key: Random key

    Returns:
        Updated model, optimizer state, model state and loss.
    """
    # Compute the loss in regards to the model parameters.
    (loss, (model_state, _)), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, data, model_state, key
    )
    # Update the model parameters.
    updates, opt_state = optim.update(grad, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    return model, opt_state, model_state, loss


def train(
    model: Model,
    optim: GradientTransformation,
    opt_state: OptState,
    model_state: eqx.nn.State,
    train_dataloader: grain.DataLoader,
    validation_dataloader: grain.DataLoader,
    loss_fn: Callable,
    key: PRNGKeyArray,
    log_rate: int,
) -> tuple[Model, eqx.nn.State]:
    """Train a model.

    Args:
        model: Model to train.
        optim: Optimizer to use.
        opt_state: State of the optimizer.
        model_state: Batch norm state.
        train_dataloader: DataLoader for the training dataset.
        validation_dataloader: DataLoader for the validation dataset.
        loss_fn: Loss function to use.
        key: Random key
        log_rate: Frequency at which to log the training process.

    Returns:
        Trained model, model state and losses.
    """
    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    @partial(eqx.filter_jit, donate="all")
    def scan_step(carry, it):
        (dynamic_model, opt_state, model_state), (data, step_key) = carry, it
        model, opt_state, model_state, loss = step(
            eqx.combine(dynamic_model, static_model), data, opt_state, optim, model_state, loss_fn, step_key
        )
        return (eqx.filter(model, eqx.is_array), opt_state, model_state), loss

    for pack_step, (data, step_key) in enumerate(zip(train_dataloader, infinite_rng_split(key))):
        data = AudioDataset(*data)
        print(data)
        carry, it = (dynamic_model, opt_state, model_state), (data, split(step_key, len(data.mos)))
        (dynamic_model, opt_state, model_state), loss = lax.scan(scan_step, carry, it)

        log.log_multiple(loss, "train", "loss")

        if pack_step % log_rate == 0:
            model = eqx.combine(dynamic_model, static_model)
            log.log_eval(model, model_state, validation_dataloader, step_key, loss_fn)
            print(f"Step: {pack_step}, Loss: {loss.mean()}")
            log.save_model(model, model_state, pack_step)
    return model, model_state
