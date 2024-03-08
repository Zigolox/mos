"""Module responsible for logging the training process using Weights and Biases."""

from typing import Callable

import equinox as eqx
import grain.python as grain
from eval import evaluate
from jaxtyping import Array, PRNGKeyArray

import wandb
from mos.datasetv2 import REPO_ROOT
from mos.models import Model


MODEL_SAVE_PATH = REPO_ROOT / "models"


def log_eval(
    model: Model,
    model_state: eqx.nn.State,
    dataloader: grain.DataLoader,
    key: PRNGKeyArray,
    loss_fn: Callable,
) -> None:
    """Log the evaluation of the model using Weights and Biases.

    Args:
        model: Model to evaluate.
        model_state: State of the model.
        dataloader: Dataloader to use.
        key: Key to use for randomness.
        loss_fn: Loss function to use.
    """
    # Compute the loss in regards to the model parameters.
    loss, spearmann, pearson = evaluate(model, model_state, dataloader, loss_fn, key)
    print(f"Loss: {loss:.4f}, Spearmann: {spearmann:.4f}, Pearson: {pearson:.4f}")
    # Log the loss.
    wandb.log({"eval": {"loss": loss, "spearmann": spearmann, "pearson": pearson}})


def log_multiple(array: Array, tag: str, name: str) -> None:
    """Log multiple values of an array using Weights and Biases."""
    for i in range(array.shape[0]):
        wandb.log({tag: {name: array[i]}})


def save_model(model, model_state, epoch):
    """Saves the model to wandb."""
    model_file_name = f"models/{model.__class__.__name__}_gstep{epoch}.eqx"
    state_file_name = f"models/{model.__class__.__name__}_state_gstep{epoch}.eqx"

    # Create directory if it does not exist
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(model_file_name, model)
    eqx.tree_serialise_leaves(state_file_name, model_state)
    wandb.save(model_file_name)
    wandb.save(state_file_name)
