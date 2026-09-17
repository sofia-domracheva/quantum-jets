import hashlib
import json
from dataclasses import dataclass, asdict, field

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

@dataclass(frozen=True)
class RuntimeConfig:
    runs_dir: str = "runs"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.log_level not in LOG_LEVELS:
            raise ValueError("log_level must be one of " + ", ".join(sorted(LOG_LEVELS)))

@dataclass(frozen=True)
class RunConfig:
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)

    def config_hash(self) -> str:
        data = asdict(self)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:8]

