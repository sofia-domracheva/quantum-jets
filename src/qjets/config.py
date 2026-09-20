import hashlib
import json
import numpy as np
from dataclasses import dataclass, asdict, field

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

@dataclass(frozen=True)
class RuntimeConfig:
    runs_dir: str = "runs"
    log_level: str = "INFO"
    step_size: int = 1000

    def __post_init__(self) -> None:
        if self.log_level not in LOG_LEVELS:
            raise ValueError("log_level must be one of " + ", ".join(sorted(LOG_LEVELS)))

        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

@dataclass(frozen=True)
class DataConfig:
    top_k: int = 3
    resp_low: float = 0.9
    resp_high: float = 1.1
    n_samples: int = 1000
    path: str = "mc_jets-2020.part01.root"
    tree_name: str = "JetRecoTree"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.resp_low <= 0 or self.resp_high <= 0:
            raise ValueError("resp_low and resp_high must be positive")

        if self.resp_low >= self.resp_high:
            raise ValueError("resp_low must be less than resp_high")

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if self.n_samples % 2 != 0:
            raise ValueError("n_samples must be even")

        if self.path == "":
            raise ValueError("path must be non-empty")

        if self.tree_name == "":
            raise ValueError("tree_name must be non-empty")

@dataclass(frozen=True)
class ModelConfig:
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    gamma_grid: tuple[float | str, ...] = ("scale", 0.01, 0.1, 1.0)
    cv_folds: int = 5
    scoring: str = "roc_auc"

    def __post_init__(self) -> None:
        if len(self.c_grid) == 0:
            raise ValueError("c_grid must be non-empty")

        if len(self.gamma_grid) == 0:
            raise ValueError("gamma_grid must be non-empty")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be positive and minimum is 2")

@dataclass(frozen=True)
class PreprocessingConfig:
    test_size: float = 0.2
    qubit_dims: tuple[int, ...] = (4, 6, 8)
    encoding_range: tuple[float, float] = (0.0, 2.0)

    def __post_init__(self) -> None:
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")

        if len(self.qubit_dims) == 0:
            raise ValueError("qubit_dims must be non-empty")

        if any(d <= 0 for d in self.qubit_dims):
            raise ValueError("qubit_dims must be positive")

    def encoding_range_rad(self) -> tuple[float, float]:
        return (self.encoding_range[0] * np.pi, self.encoding_range[1] * np.pi)

@dataclass(frozen=True)
class RunConfig:
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prep: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def config_hash(self) -> str:
        data = asdict(self)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:8]

