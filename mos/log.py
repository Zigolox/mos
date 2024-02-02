"""Module responsible for logging the training process using Weights and Biases."""
from typing import Callable

import equinox as eqx
from dataset import AudioDataset
from eval import evaluate
from jaxtyping import Array, PRNGKeyArray

import wandb
from models import DeepMos


def log_eval(
    model: DeepMos,
    data: AudioDataset,
    model_state: eqx.nn.State,
    key: PRNGKeyArray,
    loss_fn: Callable,
    tag: str,
) -> None:
    # Compute the loss in regards to the model parameters.
    loss, spearmann, pearson = evaluate(model, model_state, data, loss_fn, key)

    # Log the loss.
    wandb.log({tag: {"loss": loss, "spearmann": spearmann, "pearson": pearson}})


def log_multiple(array: Array, tag: str, name: str) -> None:
    for i in range(array.shape[0]):
        wandb.log({tag: {name: array[i]}})


def save_model(model, model_state, epoch):
    """Saves the model to wandb"""
    model_file_name = f"models/{model.__class__.__name__}_gstep{epoch}.eqx"
    state_file_name = f"models/{model.__class__.__name__}_state_gstep{epoch}.eqx"

    eqx.tree_serialise_leaves(model_file_name, model)
    eqx.tree_serialise_leaves(state_file_name, model_state)
    wandb.save(model_file_name)
    wandb.save(state_file_name)
