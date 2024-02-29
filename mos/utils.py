"""Utility functions."""

from collections.abc import Iterator

from jax import Array, random


def infinite_rng_split(key: Array) -> Iterator[Array]:
    """Infinite random number generator."""
    while True:
        key, subkey = random.split(key)
        yield subkey
