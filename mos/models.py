"""All models used in the project."""
import equinox as eqx
from einops import rearrange, repeat
from jax import lax, numpy as jnp
from jax.nn import elu, relu
from jax.random import split
from jaxtyping import Array, Float, PRNGKeyArray


# TODO: Add type aliases for hidden states
hidden_state_lstm = tuple[Float[Array, " hidden"], Float[Array, " hidden"]]


class LSTM(eqx.Module):
    """LSTM model."""

    cell: eqx.nn.LSTMCell
    reverse: bool

    def __init__(self, input_size: int, hidden_size: int, *, reverse: bool = False, key: PRNGKeyArray) -> None:
        """LSTM."""
        self.cell = eqx.nn.LSTMCell(input_size, hidden_size, key=key)
        self.reverse = reverse

    def __call__(self, inputs: Float[Array, "time feature"]) -> Float[Array, "time hidden"]:
        """Forward pass of the LSTM.

        Args:
            inputs: Input sequence of shape (time, feature_size).

        Returns:
            Hidden state of shape (time, hidden_size).
        """
        # Create initial state
        init_state: hidden_state_lstm = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

        # Define scan function using the LSTM cell
        def scan_fn(
            state: hidden_state_lstm, x: Float[Array, " input"]
        ) -> tuple[hidden_state_lstm, hidden_state_lstm]:
            hidden: hidden_state_lstm = self.cell(x, state)
            return hidden, hidden

        # Run scan function over the input sequence and return only the hidden state and not the memory state
        _, (_, outputs) = lax.scan(scan_fn, init_state, inputs, reverse=self.reverse)

        return outputs


class BiLSTM(eqx.Module):
    """Bi-Directional LSTM consisting of two LSTMs,one forward and one backward which outputs are concatenated."""

    forward_lstm: LSTM
    backward_lstm: LSTM

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray) -> None:
        """Initialize the BiLSTM.

        Args:
            input_size: Size of the input vector.
            hidden_size: Size of the hidden state.
            key: Random key.
        """
        self.forward_lstm = LSTM(input_size, hidden_size, reverse=False, key=key)
        self.backward_lstm = LSTM(input_size, hidden_size, reverse=True, key=key)

    def __call__(
        self, inputs: Float[Array, "time feature"], *, key: PRNGKeyArray
    ) -> Float[Array, "time 2*hidden"]:
        """Forward pass of the BiLSTM."""
        del key

        forward_states = self.forward_lstm(inputs)  # ´TODO: Check if vmap increases performance
        backward_states = self.backward_lstm(inputs)

        # Combine two axes
        combined = jnp.concatenate([forward_states, backward_states], axis=1)
        return combined


class DropNormActUnit(eqx.nn.Sequential):
    """Dropout, BatchNorm, ReLU."""

    def __init__(self, *, input_size: int):
        """Initialize network unit with dropout, batchnorm and activation function."""
        super().__init__([eqx.nn.Dropout(0.3), eqx.nn.BatchNorm(input_size, "batch"), eqx.nn.Lambda(relu)])


class DeePMOSConvBlock(eqx.nn.Sequential):
    """Convolutional block of the DeePMOS architecture."""

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        """Initialize the convoluational block."""
        keys = split(key, 3)
        super().__init__(
            [
                eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=keys[0]),
                eqx.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, key=keys[1]),
                eqx.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=(1, 3), key=keys[2]),
            ]
        )


class ConvEncoder(eqx.nn.Sequential):
    """Convolutional encoder."""

    def __init__(self, *, key: PRNGKeyArray):
        """Initialize the convolutional encoder."""
        keys = split(key, 4)
        super().__init__(
            [
                eqx.nn.Lambda(lambda x: rearrange(x, "time feature -> 1 time feature")),
                DeePMOSConvBlock(in_channels=1, out_channels=16, key=keys[0]),
                DropNormActUnit(input_size=16),
                DeePMOSConvBlock(in_channels=16, out_channels=32, key=keys[1]),
                DropNormActUnit(input_size=32),
                DeePMOSConvBlock(in_channels=32, out_channels=64, key=keys[2]),
                DropNormActUnit(input_size=64),
                DeePMOSConvBlock(in_channels=64, out_channels=128, key=keys[3]),
                DropNormActUnit(input_size=128),
            ]
        )


class DeePMOSLSTMBlock(eqx.nn.Sequential):
    """LSTM block of the DeePMOS architecture."""

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        """Initialize the LSTM block."""
        super().__init__(
            [
                eqx.nn.Lambda(lambda x: rearrange(x, "c time w -> time (c w)")),
                BiLSTM(input_size=input_size, hidden_size=hidden_size, key=key),
            ]
        )


class DeePMOSEncoder(eqx.nn.Sequential):
    """Enconder part of the DeePMOS architecture."""

    def __init__(self, *, key: PRNGKeyArray):
        """Initialzize the encoder part."""
        conv_key, lstm_key = split(key, 2)
        super().__init__(
            [
                ConvEncoder(key=conv_key),
                DeePMOSLSTMBlock(input_size=512, hidden_size=128, key=lstm_key),
            ]
        )


class DeePMOSVarianceDecoder(eqx.nn.Sequential):
    """Variance Decoder of the DeePMOS architecture."""

    def __init__(self, *, key: PRNGKeyArray):
        """Initialize the variance decoder, it predict the positive variance."""
        keys = split(key, 2)
        super().__init__(
            [
                eqx.nn.Linear(256, 128, key=keys[0]),
                eqx.nn.Lambda(relu),
                eqx.nn.Dropout(0.3),
                eqx.nn.Linear(128, 1, key=keys[1]),
                eqx.nn.Lambda(lambda x: elu(x) + 1),
            ]
        )


class DeePMOSMeanDecoder(eqx.nn.Sequential):
    """Mean Decoder of the DeePMOS architecture."""

    def __init__(self, *, key: PRNGKeyArray):
        """Initializes the mean."""
        keys = split(key, 2)
        super().__init__(
            [
                eqx.nn.Linear(256, 128, key=keys[0]),
                eqx.nn.Lambda(relu),
                eqx.nn.Dropout(0.3),
                eqx.nn.Linear(128, 1, key=keys[1]),
            ]
        )


class DeepMos(eqx.Module):
    """DeePMOS architecture."""

    encoder: DeePMOSEncoder
    mean_decoder: DeePMOSMeanDecoder
    variance_decoder: DeePMOSVarianceDecoder

    def __init__(self, *, key: PRNGKeyArray):
        """Initializes the DeepMOS network consisting of an encoder and two decoders."""
        keys = split(key, 3)
        self.encoder = DeePMOSEncoder(key=keys[0])
        self.mean_decoder = DeePMOSMeanDecoder(key=keys[1])
        self.variance_decoder = DeePMOSVarianceDecoder(key=keys[2])

    def __call__(
        self, inputs: Float[Array, "time feature"], model_state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[tuple[Float[Array, "time 1"], Float[Array, "time 1"]], eqx.nn.State]:
        """Forward pass of the DeePMOS architecture.

        Args:
            inputs: Input sequence of shape (time, feature_size).
            model_state: The batch state of the model.
            key: Random key.

        Returns:
            Mean and variance of shape (time, 1).
        """
        encoder_key, mean_key, variance_key = split(key, 3)

        # Run the encoder
        hidden, model_state = self.encoder(inputs, state=model_state, key=encoder_key)

        # Run the decoders
        mean = eqx.filter_vmap(lambda x, key: self.mean_decoder(x, key=key))(hidden, split(mean_key, len(inputs)))
        variance = eqx.filter_vmap(lambda x, key: self.variance_decoder(x, key=key))(
            hidden, split(variance_key, len(inputs))
        )
        return (mean, variance), model_state


class MultiEncoderMos(eqx.Module):
    """MultiEncoderMos architecture.

    This architecture consists of two encoders and one decoder.
    The first encoder takes in the reference signal and the second encoder takes in the degraded signal.
    """

    conv: DeePMOSConvBlock
    encoder_ref: ConvEncoder
    encoder_deg: ConvEncoder
    lstm: BiLSTM
    mean_decoder: DeePMOSMeanDecoder

    def __init__(self, *, key: PRNGKeyArray):
        """Initializes the MultiEncoderMos network consisting of two encoder and one decoders."""
        keys = split(key, 4)
        self.conv = DeePMOSConvBlock(in_channels=1, out_channels=16, key=keys[0])
        self.encoder_ref = ConvEncoder(key=keys[0])
        self.encoder_deg = ConvEncoder(key=keys[1])
        self.lstm = BiLSTM(input_size=1024, hidden_size=128, key=keys[2])
        self.mean_decoder = DeePMOSMeanDecoder(key=keys[3])

    def __call__(
        self,
        inputs_ref: Float[Array, "time feature"],
        inputs_deg: Float[Array, "time feature"],
        model_state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "time 1"], eqx.nn.State]:
        """Forward pass of the MultiEncoderMos architecture.

        Args:
            inputs_ref: Input sequence of shape (time, feature_size).
            inputs_deg: Input sequence of shape (time, feature_size).
            model_state: The batch state of the model.
            key: Random key.

        Returns:
            Mean of shape (time, 1).
            The batch state of the model.
        """
        encoder_ref_key, encoder_deg_key, lstm_key, mean_key = split(key, 4)

        # Run the encoder
        # hidden_ref, model_state = self.encoder_ref(inputs_ref, state=model_state, key=encoder_ref_key)
        # hidden_deg, model_state = self.encoder_deg(inputs_deg, state=model_state, key=encoder_deg_key)

        # Run the decoders
        # hidden = rearrange([hidden_ref, hidden_deg], "two channel time feature -> time (two channel feature)")
        # hidden = self.lstm(hidden, key=lstm_key)

        # mean = eqx.filter_vmap(lambda x, key: self.mean_decoder(x, key=key))(
        #    hidden, split(mean_key, len(inputs_ref))
        # )
        inputs_ref = rearrange(inputs_ref, "time feature -> 1 time feature")
        hidden_ref = self.conv(inputs_ref)
        mean = jnp.repeat(hidden_ref.mean(), inputs_ref.shape[0])
        print(mean.shape)
        return mean, model_state
