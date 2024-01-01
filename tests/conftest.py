import jax
import pytest
from hypothesis import settings


jax.config.update("jax_numpy_dtype_promotion", "strict")
jax.config.update("jax_numpy_rank_promotion", "raise")

settings.register_profile("fast", max_examples=5, deadline=None)


@pytest.fixture
def getkey():
    # Delayed import so that jaxtyping can transform the AST of Equinox before it is
    # imported, but conftest.py is ran before then.
    import equinox.internal as eqxi

    return eqxi.GetKey()
