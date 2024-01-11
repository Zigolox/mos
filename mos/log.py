"""Module responsible for logging the training process using Weights and Biases."""
import wandb
from eval import evaluate
from jaxtyping import PRNGKeyArray, Array
from typing import Callable
from dataset import AudioDataset
from models import DeepMos
import equinox as eqx


def log(
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
    model_file_name = f"{model.__class__.__name__}_gstep{epoch}.eqx"
    model = eqx.nn.inference_mode(model)
    model = eqx.Partial(model, state=model_state)

    eqx.tree_serialise_leaves(model_file_name, model)
    wandb.save(model_file_name)
