import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qjets.config import PreprocessingConfig
log = logging.getLogger(__name__)
TOL = 1e-9

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def fit_transformers(X_train: np.ndarray, max_dim: int, seed: int) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=max_dim, random_state=seed))]).fit(X_train)

def fit_angle_scaler(X_train_slice: np.ndarray, encoding_range: tuple[float, float]) -> MinMaxScaler:
    return MinMaxScaler(feature_range=encoding_range).fit(X_train_slice)

def to_angles(scaler: MinMaxScaler, X: np.ndarray, encoding_range: tuple[float, float]) -> tuple[np.ndarray, float]: 
    X_proj = scaler.transform(X)
    low, high = encoding_range
    mask = (X_proj < low - TOL) | (X_proj > high + TOL)
    
    return np.clip(X_proj, low, high), mask.mean()

def slice_dims(X_proj: np.ndarray, n: int) -> np.ndarray:
    return X_proj[:, :n]

def prepare_all(X: np.ndarray, y: np.ndarray, config: PreprocessingConfig, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = split_train_test(X, y, config.test_size, seed)
    train_pipeline = fit_transformers(X_train, max_dim=max(config.qubit_dims), seed=seed)

    transformed_train_pipeline = train_pipeline.transform(X_train)
    transformed_test_pipeline = train_pipeline.transform(X_test)
    low_high = config.encoding_range_rad()
    explained = train_pipeline.named_steps["pca"].explained_variance_ratio_
    result = {}
    for q in config.qubit_dims:
        sliced_train = slice_dims(transformed_train_pipeline, q)
        sliced_test = slice_dims(transformed_test_pipeline, q)
        minmax_train = fit_angle_scaler(sliced_train, low_high)

        train_ang, frac_train = to_angles(minmax_train, sliced_train, low_high)
        test_ang, frac_test = to_angles(minmax_train, sliced_test, low_high)
        result[q] = {"train":train_ang, "test":test_ang}

        explained_var = explained[:q].sum()
        log.info(f"Qubit {q}: {frac_train:.2%} train, {frac_test:.2%} test, explained var: {explained_var:.2%}")
    return result, y_train, y_test