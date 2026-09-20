import numpy as np
import uproot
import awkward as ak
from pathlib import Path
from collections.abc import Iterator
from qjets.data.features import READ_BRANCHES, filter_chunk, build_features, build_labels

def read_branches(path: Path, name: str, branches: list[str], step_size: int = 1000) -> Iterator[ak.Array]:
    with uproot.open(path) as f:
        tree = f[name]

        for chunk in tree.iterate(expressions=branches, step_size=step_size, library="ak"):
            yield chunk

def load_jets_root(path: Path, name: str, top_k: int, resp_low: float, resp_high: float, step_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for chunk in read_branches(path, name, list(READ_BRANCHES), step_size):
        chunk = filter_chunk(chunk, top_k)
        features.append(build_features(chunk, top_k))
        labels.append(build_labels(chunk, resp_low, resp_high))

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)

    return X, y