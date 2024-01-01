"""Module responsible for the training loop."""

import equinox as eqx
import jax.numpy as jnp
from dataset import AudioDataset, VCC18Dataset
from jax import lax
from jax.random import split
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, PRNGKeyArray
from loss import batch_loss
from models import DeepMos
from optax import GradientTransformation, OptState


def step(
    model: DeepMos,
    data: AudioDataset,
    opt_state: OptState,
    optim: GradientTransformation,
    model_state: eqx.nn.State,
    key: PRNGKeyArray,
):
    """Perform a single training step.

    Args:
        model: Model to train.
        data: Data to use.
        opt_state: State of the optimizer.
        optim: Optimizer to use.
        model_state: Batch norm state.
        key: Random key

    Returns:
        Updated model, optimizer state, model state and loss.
    """
    # Compute the loss in regards to the model parameters.
    (loss, model_state), grad = eqx.filter_value_and_grad(batch_loss, has_aux=True)(model, data, model_state, key)
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
    model_state: eqx.nn.State,
    dataset: VCC18Dataset,
    key: PRNGKeyArray,
) -> tuple[DeepMos, eqx.nn.State, list[Float[Array, " data_size//batch_size"]]]:
    """Train a model.

    Args:
        model: Model to train.
        optim: Optimizer to use.
        opt_state: State of the optimizer.
        state: State of the model.
        epochs: Number of epochs to train.
        batch_size: Size of the batch.
        model_state: State of the batch norm.
        dataset: Class dataset from dataset module
        key: Random key

    Returns:
        Trained model, model state and losses.
    """
    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    @scan_tqdm(len(dataset) // batch_size)
    def scan_step(carry, it):
        (dynamic_model, opt_state, model_state), (_, data, key) = carry, it
        model, opt_state, model_state, loss = step(
            eqx.combine(dynamic_model, static_model), data, opt_state, optim, model_state, key
        )
        return (eqx.filter(model, eqx.is_array), opt_state, model_state), loss

    losses = []
    carry = (dynamic_model, opt_state, model_state)
    epoch_keys = split(key, epochs)

    for key in epoch_keys:
        data = dataset.scan_all(batch_size, key=key)
        it = (jnp.arange(len(data.wav)), data, split(key, len(data.wav)))
        (dynamic_model, opt_state, model_state), loss = lax.scan(scan_step, carry, it)
        print(loss)
        losses.append(loss)

    model = eqx.combine(dynamic_model, static_model)

    return model, model_state, losses


if __name__ == "__main__":
    import argparse
    import pathlib

    import jax
    from optax import adamw

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model for.")
    parser.add_argument("--batch-size", type=int, default=2, help="Size of the batch to use for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use for training.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=pathlib.Path(__file__).parent.parent / "data" / "vcc2018",
        help="Directory where the data is stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the random number generator.")

    args = parser.parse_args()

    # Set the seed
    key = jax.random.key(args.seed)

    # Load the dataset
    dataset = VCC18Dataset(args.data_dir, args.data_dir / "small_vcc2018_testing_data.csv")

    # Create the model
    model, model_state = eqx.nn.make_with_state(DeepMos)(key=key)

    # Create the optimizer
    optim = adamw(learning_rate=args.lr)

    # Initialize the optimizer
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Train the model
    model, model_state, losses = train(
        model,
        optim,
        opt_state,
        args.epochs,
        args.batch_size,
        model_state,
        dataset,
        key,
    )

    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=model_state)
