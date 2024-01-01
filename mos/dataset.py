"""Dataset module for currently the VCC18 dataset."""

from collections.abc import Iterator
from pathlib import Path
from typing import Union

import equinox as eqx
import librosa
import pandas as pd
from einops import rearrange
from jax import numpy as jnp, random, tree_map, lax, devices, device_put
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm


def pad_batch(wavs: list[Float[Array, "_ feature"]]) -> Float[Array, "batch time feature"]:
    """Pad a batch of wavs to the same length."""
    max_len = max(wav.shape[0] for wav in wavs)
    return jnp.stack([lax.pad(wav, 0.0, ((0, max_len - wav.shape[0], 0), (0, 0, 0))) for wav in wavs])


class AudioDataset(eqx.Module):
    """Dataset for audio files."""

    wav: Float[Array, " batch time feature"] = eqx.field(converter=pad_batch)
    mean: Float[Array, " batch"] = eqx.field(converter=jnp.asarray)
    score: Int[Array, " batch"] = eqx.field(converter=jnp.asarray)
    judge_id: Int[Array, " batch"] = eqx.field(converter=jnp.asarray)


class VCC18Dataset:
    """Dataset for the VCC18 dataset."""

    def __init__(self, data_path: Path, score_csv_path: Path, device: str = "cpu"):
        """Prepares the dataset by preprocessing it for training or validation.

        Args:
            data_path: Path to the data folder.
            score_csv_path: Path to the score csv file.

        """

        self.device = devices(device)[0]
        # Read score csv
        self.scores = pd.read_csv(
            score_csv_path,
            index_col=False,
            dtype={"JUDGE": "category", "WAV_PATH": str, "MOS": int, "MEAN": float},
        )

        # Create judge id translation table
        self.str2id = {judge_id: i for i, judge_id in enumerate(sorted(self.scores["JUDGE"].unique()))}
        self.scores["JUDGE"].replace(self.str2id, inplace=True)

        # Turn all wav files into spectrograms and store them in a dictionary
        self.wavs = {
            wav_file: self._generate_wav(data_path / wav_file)
            for wav_file in tqdm(self.scores["WAV_PATH"].unique(), desc="Generating spectrograms")
        }

        # Get all systems
        self.systems = self.scores["WAV_PATH"].apply(lambda x: x.split("_")[0]).unique()

    def _generate_wav(self, wav_path: Path) -> Float[Array, "time feature"]:
        """Generate a spectrogram from a wav file."""
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = jnp.abs(librosa.stft(wav, n_fft=512)).T
        return wav

    def __getitem__(self, idx: Union[int, Int[Array, " batch"]]) -> AudioDataset:
        """Get a sample or samples from the dataset.

        Args:
            idx: Index of the sample to get.

        Returns:
            wav: The wave file as a numpy array.
            mean: The mean score of the sample.
            score: The score of the sample.
            judge_id: The judge id of the sample.
        """
        rows = self.scores.iloc[idx]
        if isinstance(idx, int):
            wav = [self.wavs[rows["WAV_PATH"]]]
        else:
            wav = [self.wavs[row["WAV_PATH"]] for _, row in rows.iterrows()]
        return AudioDataset(wav, rows["MEAN"], rows["MOS"], rows["JUDGE"])

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.scores)

    def epoch(self, batch_size: int, *, key: PRNGKeyArray) -> Iterator[AudioDataset]:
        """Create an iterator over the dataset.

        Args:
            batch_size: Size of the batch.
            key: Key to use for shuffling.

        Returns:
            Iterator yielding batches of the dataset.
        """
        # Permute the dataset
        perm = random.permutation(key, len(self))

        for indices in jnp.split(perm, len(self) // batch_size):
            # Get the samples of the batch
            yield self[indices]

    def scan_all(self, batch_size: int, scan_size: int, *, key: PRNGKeyArray) -> Iterator[AudioDataset]:
        """Get a random sample from the dataset.

        Args:
            batch_size: Size of the batch.
            scan_size: Size of the scan iterations.
            key: Key to use for shuffling.

        Returns:
            wav: The wave file as a numpy array.
            mean: The mean score of the sample.
            score: The score of the sample.
            judge_id: The judge id of the sample.
        """

        perm = random.permutation(key, len(self))[: len(self) - len(self) % (scan_size * batch_size)]
        for p in jnp.split(perm, scan_size):
            data = self[p]
            n_scan = len(data.mean) // batch_size
            yield tree_map(lambda x: rearrange(x, "(s b) ... -> s b ...", s=n_scan, b=batch_size), data)
