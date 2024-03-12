"""Grain dataloader for the NISQA dataset."""

import dataclasses
from pathlib import Path
from typing import Union

import grain.python as grain
import librosa
import numpy as np
import pandas as pd

from mos.audiodataset import AudioDataset


REPO_ROOT = Path(__file__).resolve().parent.parent


class NISQADataset(grain.RandomAccessDataSource):
    """Loader and server of the NISQA dataset."""

    def __init__(
        self,
        data_path: Path,
        score_csv_path: Path,
        data_type="NISQA_TRAIN_SIM",
        size=-1,
        only_deg=False,
        only_ref=False,
        noise_amplitude=0.0,
    ):
        """Prepares the dataset by preprocessing it for training or validation.

        Args:
            data_path: Path to the data folder.
            score_csv_path: Path to the score csv file.
            data_type: Type of the data to load.
            size: Size of the dataset to load.
        """
        # Read score csv
        frame: pd.DataFrame = pd.read_csv(
            score_csv_path,
            index_col=False,
        )
        self.scores = frame[frame["db"] == data_type].reset_index(drop=True)
        self.data_path = data_path
        if size > 0:
            self.scores = self.scores[:size]
        self.filepath_ref = "filepath_deg" if only_deg else "filepath_ref"
        self.filepath_deg = "filepath_ref" if only_ref else "filepath_deg"
        self.noise_amplitude = noise_amplitude
        self.rng = np.random.default_rng()

    def _generate_wav(self, wav_path: Path, file_type: str) -> np.ndarray:
        """Generate a spectrogram from a wav file."""
        wav, _ = librosa.load(wav_path, sr=16000)
        if file_type == "ref":
            wav += np.random.default_rng(abs(hash(wav_path))).normal(0, self.noise_amplitude, wav.shape)
        return np.abs(librosa.stft(wav, n_fft=512)).T

    def _load_wav(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Load a wave file from the dataset.

        Args:
            row: The row of the dataset to load.

        Returns:
            ref: The reference wave file as a numpy array.
            deg: The degraded wave file as a numpy array.
        """
        ref = self._generate_wav(self.data_path / row[self.filepath_ref], "ref")
        deg = self._generate_wav(self.data_path / row[self.filepath_deg], "deg")
        return ref, deg

    def __getitem__(self, idx: Union[int, slice]) -> tuple[np.ndarray, np.ndarray, float]:
        """Get a sample or samples from the dataset.

        Args:
            idx: Index of the sample to get.

        Returns:
            wav: The wave file as a numpy array.
            score: The score of the sample.
        """
        rows = self.scores.iloc[idx]  # type: ignore
        ref, deg = self._load_wav(rows)
        return ref, deg, rows["mos"]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.scores)


@dataclasses.dataclass(frozen=True)
class PadTransform(grain.MapTransform):
    """Pad a spectrogram to a given length."""

    length: int

    def _pad(self, element: np.ndarray) -> np.ndarray:
        """Pad the spectrogram to the given length."""
        if element.shape[0] < self.length:
            return np.pad(element, ((0, self.length - element.shape[0]), (0, 0)))
        return element[: self.length]

    def map(self, element: tuple) -> tuple:  # type: ignore
        """Pad the spectrogram to the given length."""
        return tuple(self._pad(e) if isinstance(e, np.ndarray) else e for e in element)


class AudioDatasetTransform(grain.MapTransform):
    """Convert the Tuple into a equinox module."""

    def map(self, element: tuple) -> AudioDataset:  # type: ignore
        """Map over the tuple and convert it into a AudioDataset."""
        ref, deg, mos = element
        return AudioDataset(ref=ref, deg=deg, mos=mos)


if __name__ == "__main__":
    import time

    dataset = NISQADataset(
        REPO_ROOT / "data" / "nisqa",
        REPO_ROOT / "data" / "nisqa" / "NISQA_corpus_file.csv",
        data_type="NISQA_VAL_SIM",
        size=100,
        noise_amplitude=0.0,
    )
    read_options = grain.ReadOptions(num_threads=16, prefetch_buffer_size=100)
    a = grain.load(
        dataset,
        num_epochs=2,
        transformations=[
            PadTransform(1000),
        ],
        shuffle=True,
        seed=0,
        worker_count=1,
        read_options=read_options,
    )
    start_time = time.time()
    for i in a:
        print("sleeping", i.mos)
        time.sleep(3)
        print(time.time() - start_time)
        start_time = time.time()

    for i in a:
        print("asdasd", i.mos)
        time.sleep(5)
