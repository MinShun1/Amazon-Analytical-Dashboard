"""Read the raw Kaggle CSVs straight from the dataset zip (no extraction)."""
from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import pandas as pd

from src.config import KAGGLE_URL, RAW_FILES, RAW_ZIP


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset zip is missing or incomplete."""


def check_raw_zip(path: Path = RAW_ZIP) -> list[str]:
    """Return the list of problems with the raw zip (empty list means OK)."""
    if not path.exists():
        return [
            f"Dataset zip not found at '{path.relative_to(path.parents[2])}'. "
            f"Download it from {KAGGLE_URL} and save it under that name."
        ]
    with zipfile.ZipFile(path) as zf:
        names = {Path(n).name for n in zf.namelist()}
    missing = [f for f in RAW_FILES.values() if f not in names]
    return [f"Missing file inside the zip: '{m}'" for m in missing]


def _member_name(zf: zipfile.ZipFile, file_name: str) -> str:
    """Find a CSV inside the zip, whether at the root or in a sub-folder."""
    for name in zf.namelist():
        if Path(name).name == file_name:
            return name
    raise DatasetNotFoundError(f"'{file_name}' not found in {RAW_ZIP.name}")


def read_raw(path: Path = RAW_ZIP) -> dict[str, pd.DataFrame]:
    """Load the seven raw CSVs exactly as the notebook did (``pd.read_csv``
    with default arguments), keyed by the notebook's dataset names."""
    problems = check_raw_zip(path)
    if problems:
        raise DatasetNotFoundError(" ".join(problems))

    dfs: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        for key, file_name in RAW_FILES.items():
            with zf.open(_member_name(zf, file_name)) as fh:
                with warnings.catch_warnings():
                    # Same DtypeWarning the notebook shows for the unlabeled
                    # 'Unnamed: 22' column, which is dropped later.
                    warnings.simplefilter("ignore", pd.errors.DtypeWarning)
                    dfs[key] = pd.read_csv(fh)
    return dfs
