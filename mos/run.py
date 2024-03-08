"""Train the MOS model."""

import argparse
import pathlib
from functools import partial

import equinox as eqx
import grain.python as grain
import jax
from optax import adamw

import wandb
from mos.datasetv2 import NISQADataset, PadTransform, REPO_ROOT
from mos.loss import multi_head_batch_loss
from mos.models import MultiEncoderMos
from mos.train import train


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model for.")
    parser.add_argument("--batch-size", type=int, default=1, help="Size of the batch to use for training.")
    parser.add_argument("--scan-size", type=int, default=1, help="Size of the scan to use for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use for training.")
    parser.add_argument("--dataset-type", type=str, default="vcc2018", help="Dataset to use for training.")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "data" / "nisqa",
        help="Directory where the data is stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the random number generator.")
    parser.add_argument("--train_dataset", type=str, default="NISQA_corpus_file.csv")
    parser.add_argument("--validation_dataset", type=str, default="NISQA_corpus_file.csv")
    parser.add_argument("--validation_size", type=int, default=50)
    parser.add_argument("--only_deg", type=bool, default=False, help="Only use the degraded audio as input.")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb for logging.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to use for data loading.")
    parser.add_argument("--drop_remainder", type=bool, default=True, help="Drop remainder of the batch.")
    parser.add_argument("--log_rate", type=int, default=100, help="Rate at which to log the loss.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run = wandb.init(project="mos", config=vars(args), mode="online" if args.wandb else "disabled", dir=REPO_ROOT)
    wandb.config.update(args)

    # Set the seed
    key = jax.random.key(args.seed)

    # Load the dataset
    train_dataset = NISQADataset(
        args.data_dir,
        args.data_dir / args.train_dataset,
        data_type="NISQA_TRAIN_SIM",
    )
    validation_dataset = NISQADataset(
        args.data_dir,
        args.data_dir / args.validation_dataset,
        data_type="NISQA_VAL_SIM",
    )
    dataloader = partial(
        grain.load,
        transformations=(
            PadTransform(1000),
            grain.Batch(args.batch_size, drop_remainder=args.drop_remainder),
            grain.Batch(args.scan_size, drop_remainder=args.drop_remainder),
        ),
        shuffle=True,
        seed=args.seed,
        worker_count=args.workers,
    )
    train_dataloader = dataloader(train_dataset, num_epochs=args.epochs)
    validation_dataloader = dataloader(validation_dataset, num_epochs=1)

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
        model_state,
        train_dataloader,
        validation_dataloader,
        loss_fn,
        key,
        args.log_rate,
    )
