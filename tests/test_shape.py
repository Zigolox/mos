#!/usr/bin/env python3

from typing import get_args

import equinox as eqx
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from jax import numpy as jnp
from jax.random import key
from jaxtyping import Array, Float, Key

from mos.models import LSTM


settings.load_profile("fast")

st.register_type_strategy(
    Key[Array, ""],
    st.builds(key, st.integers(min_value=0, max_value=2**30)),
)

st.register_type_strategy(
    LSTM,
    st.builds(
        LSTM,
        input_size=st.integers(min_value=1, max_value=10),
        hidden_size=st.integers(min_value=1, max_value=10),
        key=st.from_type(Key[Array, ""]),
    ),
)
st.register_type_strategy(
    int,
    st.integers(min_value=1, max_value=10),
)


def create_array_strategy(array_type: type) -> st.SearchStrategy[Array]:
    dtype = get_args(array_type)[0]
    dims = len(get_args(array_type)[1].split())
    return arrays(
        dtype=dtype, shape=st.tuples(*[st.integers(min_value=1, max_value=10) for _ in range(dims)])
    ).map(lambda x: jnp.array(x))


st.register_type_strategy(
    Float[Array, "time input"],
    arrays(
        dtype=jnp.float32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10),
        ),
    ).map(lambda x: jnp.array(x)),
)


@given(...)
def test_lstm_init(input_size: int, hidden_size: int, key: Key[Array, ""]):
    lstm = LSTM(input_size, hidden_size, key=key)
    assert lstm.cell.input_size == input_size
    assert lstm.cell.hidden_size == hidden_size
    assert lstm.reverse is False


@given(...)
def test_lstm_forward(lstm: LSTM, size: int):
    x: Float[Array, "time input"] = jnp.arange(size * lstm.cell.input_size, dtype=jnp.float32).reshape(
        size, lstm.cell.input_size
    )
    output = lstm(x)
    assert output.shape == (x.shape[0], lstm.cell.hidden_size)


def test_forward_equals_backwards_LSTM(getkey):
    key = getkey()
    lstm_forward = LSTM(2, 10, key=key, reverse=False)
    x: Float[Array, "time input"] = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
    output = eqx.filter_jit(lstm_forward)(x)
    lstm_backward = LSTM(2, 10, key=key, reverse=True)
    output2 = lstm_backward(x[::-1])[::-1]
    # same shape
    assert output.shape == output2.shape
    # same values
    assert jnp.allclose(output, output2)


"""
def test_BiLSTM(getkey):
    lstm = BiLSTM(2, 10, key=getkey)
    assert lstm.forward_lstm.cell.input_size == 2
    assert lstm.forward_lstm.cell.hidden_size == 10
    assert lstm.backward_lstm.cell.input_size == 2
    assert lstm.backward_lstm.cell.hidden_size == 10
    x: Array = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
    output = lstm(x)
    assert output.shape == (10, 20)
"""
