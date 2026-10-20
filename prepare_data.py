"""Load the frozen audited snapshot without private source paths or API access."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# This deliberately absent path makes the private-original test skip explicitly.
FILES = [HERE / "private_sources" / "original_master_not_bundled.xlsx"]
MODELS = ["GPT-5.2", "Open Evidence", "Sonar-Reasoning-Pro", "Sonar"]
DOMAINS = ["Clinical Assessment", "Instrumental Assessment", "Therapy & Rehabilitation"]


def normalize(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary(value):
    if pd.isna(value):
        return np.nan
    label = str(value).strip().lower()
    return int(label == "yes") if label in ("yes", "no") else np.nan


def build_data():
    return pd.read_csv(HERE / "data/response_level.csv")
