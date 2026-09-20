import awkward as ak
import numpy as np
from numpy.random import Generator
import logging
log = logging.getLogger(__name__)

BRANCHES = {"RecoJets_R4_pt": True,
            "RecoJets_R4_eta": False,
            "RecoJets_R4_phi": False,
            "RecoJets_R4_m": True}

READ_BRANCHES = tuple(BRANCHES) + ("TruthJets_R4_pt",)

MEV_TO_GEV = 1000.0

def filter_chunk(chunk: ak.Array, top_k: int) -> ak.Array:
    pt = get_pt(chunk)
    truth = get_truth_pt(chunk)

    count_mask = (ak.num(pt) >= top_k) & (ak.num(truth) >= 1)
    chunk = chunk[count_mask]
    truth = get_truth_pt(chunk)

    truth_mask = truth[:, 0] > 0
    chunk = chunk[truth_mask]

    return chunk

def column_block(filtered_chunk: ak.Array, top_k: int, branch: str, is_mev: bool) -> np.ndarray:
    values = filtered_chunk[branch]
    values = values[:, :top_k]
    if is_mev:
        values = values / MEV_TO_GEV

    return ak.to_numpy(values)


def build_features(filtered_chunk: ak.Array, top_k: int) -> np.ndarray:
    result = []
    for branch, is_mev in BRANCHES.items():
        result.append(column_block(filtered_chunk, top_k, branch, is_mev))
    stacked = np.stack(result, axis=-1)
    reshaped = np.reshape(stacked, (stacked.shape[0], -1))

    pt = get_pt(filtered_chunk)
    njets = ak.to_numpy(ak.num(pt))
    HT = ak.to_numpy(ak.sum(pt, axis=1) / MEV_TO_GEV) 
    np_result = np.column_stack((reshaped, njets, HT))

    return np_result

def build_labels(chunk: ak.Array, resp_low: float, resp_high: float) -> np.ndarray:
    pt = get_pt(chunk)
    truth = get_truth_pt(chunk)

    response = pt[:, 0] / truth[:, 0]
    mask = (response >= resp_low) & (response <= resp_high)

    return ak.to_numpy(mask).astype(int)

def get_pt(chunk: ak.Array) -> ak.Array:
    return chunk["RecoJets_R4_pt"]

def get_truth_pt(chunk: ak.Array) -> ak.Array:
    return chunk["TruthJets_R4_pt"]

def build_sample(X: np.ndarray, y: np.ndarray, n_samples: int, rng: Generator) -> tuple[np.ndarray, np.ndarray]:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    half_samples = n_samples // 2
    idx0_count = len(idx0) 
    idx1_count = len(idx1)

    min_count = min(idx0_count, idx1_count, half_samples)
    if min_count * 2 < n_samples:
        log.warning(f"Requested {n_samples} samples, but only {min_count * 2} are available")

    idx0_sample = rng.choice(idx0, min_count, replace=False)
    idx1_sample = rng.choice(idx1, min_count, replace=False)

    idx = np.concatenate((idx0_sample, idx1_sample))
    rng.shuffle(idx)

    log.info(f"Selected {len(idx)} samples")
    return X[idx], y[idx]