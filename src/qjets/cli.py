import argparse
import json

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from qjets.config import Config, RunConfig, RuntimeConfig, LOG_LEVELS
from qjets.runtime.log import setup_logging
from qjets.runtime.seeding import set_seed
from qjets.runtime.timing import Timings
from qjets.runtime.environment import collect

def main() -> None:
    parser = argparse.ArgumentParser(description="Classical vs quantum SVM on LHC jet data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=sorted(LOG_LEVELS), help="Logging level (default: INFO)")

    args = parser.parse_args()
    runtime = RuntimeConfig(log_level=args.log_level)
    config = Config(run=RunConfig(seed=args.seed))
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runtime.runs_dir) / f"{now}-{config.config_hash()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(runtime.log_level, run_dir)
    rng = set_seed(config.run.seed)
    log.info("Running with seed %s", config.run.seed)

    timings = Timings()
    with timings.stage("environment"):
        env = collect()

    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    with (run_dir / "env.json").open("w") as f:
        json.dump(env, f, indent=2)
    
    with (run_dir / "timings.json").open("w") as f:
        json.dump(timings.as_dict(), f, indent=2)