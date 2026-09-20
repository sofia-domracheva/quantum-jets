import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

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
class RunConfig:
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def config_hash(self) -> str:
        data = asdict(self)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:8]

