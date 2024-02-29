"""Evaluation functions for DeepMOS."""

from typing import Callable

import equinox as eqx
import grain.python as grain
from jax import lax, numpy as jnp
from jaxtyping import PRNGKeyArray
from scipy.stats import spearmanr

from mos.datasetv2 import AudioDataset
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
        data = AudioDataset(*data)
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


if __name__ == "__main__":
    from pathlib import Path

    import grain.python as grain
    from jax import random

    from mos.datasetv2 import AudioDataset, NISQADataset, PadTransform
    from mos.loss import multi_head_batch_loss
    from mos.models import Model, MultiEncoderMos

    dataset = NISQADataset(
        Path(__file__).parent.parent / "data" / "nisqa_test",
        Path(__file__).parent.parent / "data" / "nisqa_test" / "NISQA_corpus_file.csv",
        data_type="NISQA_VAL_SIM",
        size=8,
        only_ref=True,
    )
    read_options = grain.ReadOptions(num_threads=16, prefetch_buffer_size=100)
    dataloader = grain.load(
        dataset,
        num_epochs=2,
        transformations=[
            PadTransform(1000),
            grain.Batch(2, drop_remainder=True),
            grain.Batch(2, drop_remainder=True),
        ],
        shuffle=True,
        seed=0,
        worker_count=1,
        read_options=read_options,
    )

    loss_fn = multi_head_batch_loss
    model_type = MultiEncoderMos
    # Create the model
    model, model_state = eqx.nn.make_with_state(model_type)(key=random.PRNGKey(0))
    print("begin evaluation")
    loss, spearmann, pearson = evaluate(model, model_state, dataloader, loss_fn, random.PRNGKey(0))
    print(f"Loss: {loss:.4f}, Spearmann: {spearmann:.4f}, Pearson: {pearson:.4f}")
