from pathlib import Path

from mos.dataset import VCC18Dataset


def test_init_VCC18Dataset():
    data_path = Path(__file__).parent.parent / "data" / "vcc2018"
    filepath = data_path / "small_vcc2018_testing_data.csv"
    dataset = VCC18Dataset(data_path, filepath)
    assert len(dataset) == 100


def test_epoch_VCC18Dataset(getkey):
    data_path = Path(__file__).parent.parent / "data" / "vcc2018"
    filepath = data_path / "small_vcc2018_testing_data.csv"
    dataset = VCC18Dataset(data_path, filepath)
    batch = next(dataset.epoch(10, key=getkey()))
    assert len(batch.wav) == 10
    assert len(batch.mean) == 10
    assert len(batch.score) == 10
    assert len(batch.judge_id) == 10


def test_scan_all_VCC18Dataset(getkey):
    data_path = Path(__file__).parent.parent / "data" / "vcc2018"
    filepath = data_path / "small_vcc2018_testing_data.csv"
    dataset = VCC18Dataset(data_path, filepath)
    batch = dataset.scan_all(10, key=getkey())
    assert batch.wav.shape[:2] == (10, 10)
    assert len(batch.mean) == 10
    assert len(batch.score) == 10
    assert len(batch.judge_id) == 10
