"""Module responsible for the training loop."""

from functools import partial

import equinox as eqx
import log
from dataset import AudioDataset, VCC18Dataset, NISQADataset
from jax import lax
from jax.random import split
from jaxtyping import Array, Float, PRNGKeyArray
from loss import batch_loss, multi_head_batch_loss
from models import DeepMos, MultiEncoderMos
from optax import GradientTransformation, OptState
from tqdm import tqdm
from typing import Callable

import wandb


def step(
    model: DeepMos,
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
    (loss, model_state), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, data, model_state, key)
    # Update the model parameters.
    updates, opt_state = optim.update(grad, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    return model, opt_state, model_state, loss


def train(
    model: DeepMos,
    optim: GradientTransformation,
    opt_state: OptState,
    epochs: int,
    batch_size: int,
    scan_size: int,
    model_state: eqx.nn.State,
    train_dataset: VCC18Dataset,
    validation_dataset: VCC18Dataset,
    validation_size: int,
    loss_fn: Callable,
    key: PRNGKeyArray,
) -> tuple[DeepMos, eqx.nn.State]:
    """Train a model.

    Args:
        model: Model to train.
        optim: Optimizer to use.
        opt_state: State of the optimizer.
        state: State of the model.
        epochs: Number of epochs to train.
        batch_size: Size of the batch.
        scan_size: Size of the scan.
        model_state: State of the batch norm.
        train_dataset: Train dataset from dataset module
        validation_dataset: Validation dataset from dataset module
        validation_size: Size of the validation dataset
        loss_fn: Loss function to use.
        key: Random key

    Returns:
        Trained model, model state and losses.
    """
    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    @partial(eqx.filter_jit, donate="all")
    def scan_step(carry, it):
        (dynamic_model, opt_state, model_state), (data, key) = carry, it
        model, opt_state, model_state, loss = step(
            eqx.combine(dynamic_model, static_model), data, opt_state, optim, model_state, loss_fn, key
        )
        return (eqx.filter(model, eqx.is_array), opt_state, model_state), loss

    for epoch, epoch_key in enumerate(split(key, epochs)):

        log.log(
            model,
            validation_dataset.get_batch(validation_size, key=epoch_key),
            model_state,
            key,
            loss_fn,
            "val",
        )
        for data, key in zip(
            train_dataset.scan_all(batch_size, scan_size, key=epoch_key), tqdm(split(epoch_key, scan_size))
        ):
            carry, it = (dynamic_model, opt_state, model_state), (data, split(key, len(data.mos)))
            (dynamic_model, opt_state, model_state), loss = lax.scan(scan_step, carry, it)
            log.log_multiple(loss, "train", "loss")

        model = eqx.combine(dynamic_model, static_model)
        log.save_model(model, model_state, epoch)
    return model, model_state


if __name__ == "__main__":
    import argparse
    import pathlib

    import jax
    from optax import adamw

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model for.")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of the batch to use for training.")
    parser.add_argument("--scan-size", type=int, default=2, help="Size of the scan to use for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use for training.")
    parser.add_argument("--dataset-type", type=str, default="vcc2018", help="Dataset to use for training.")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "data" / "vcc2018",
        help="Directory where the data is stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the random number generator.")
    parser.add_argument("--train_dataset", type=str, default="small_vcc2018_training_data.csv")
    parser.add_argument("--validation_dataset", type=str, default="small_vcc2018_val_data.csv")
    parser.add_argument("--validation_size", type=int, default=50)

    args = parser.parse_args()

    run = wandb.init(project="mos")
    wandb.config.update(args)

    # Set the seed
    key = jax.random.key(args.seed)

    # Load the dataset
    if args.dataset_type == "vcc2018":
        train_dataset = VCC18Dataset(args.data_dir, args.data_dir / args.train_dataset)
        validation_dataset = VCC18Dataset(args.data_dir, args.data_dir / args.validation_dataset)
        loss_fn = batch_loss
        model_type = DeepMos
    else:
        train_dataset = NISQADataset(args.data_dir, args.data_dir / args.train_dataset, data_type="NISQA_VAL_SIM")
        validation_dataset = NISQADataset(
            args.data_dir, args.data_dir / args.train_dataset, data_type="NISQA_VAL_SIM"
        )
        loss_fn = multi_head_batch_loss
        model_type = MultiEncoderMos
    # Create the model
    model, model_state = eqx.nn.make_with_state(model_type)(key=key)

    # Create the optimizer
    optim = adamw(learning_rate=args.lr)

    # Initialize the optimizer
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Train the model
    model, model_state = train(
        model,
        optim,
        opt_state,
        args.epochs,
        args.batch_size,
        args.scan_size,
        model_state,
        train_dataset,
        validation_dataset,
        args.validation_size,
        loss_fn,
        key,
    )

    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=model_state)
