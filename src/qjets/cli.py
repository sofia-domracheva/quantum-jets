import argparse
import json
import logging

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from numpy.random import Generator
from qjets.config import Config, DataConfig, RunConfig, RuntimeConfig, LOG_LEVELS
from qjets.runtime.log import setup_logging
from qjets.runtime.seeding import set_seed
from qjets.runtime.timing import Timings
from qjets.runtime.environment import collect
from qjets.data.features import build_sample
from qjets.data.root_io import load_jets_root

def main() -> None:
    parser = argparse.ArgumentParser(description="Classical vs quantum SVM on LHC jet data", prog="qjets")
    parser = add_data_arguments(parser)
    parser = add_runtime_arguments(parser)

    args = parser.parse_args()
    runtime = RuntimeConfig(log_level=args.log_level, step_size=args.step_size, runs_dir=args.runs_dir)
    dataconfig = DataConfig(top_k=args.top_k, resp_low=args.resp_low, resp_high=args.resp_high, n_samples=args.n_samples, path=args.path, tree_name=args.tree_name)
    config = Config(run=RunConfig(seed=args.seed), data=dataconfig)

    run_dir = create_run_dir(runtime, config)
    log = setup_logging(runtime.log_level, run_dir)
    rng = create_rng(config, log)

    timings = Timings()

    with timings.stage("load"):
        X, y = load_jets_root(Path(config.data.path), config.data.tree_name, config.data.top_k, config.data.resp_low, config.data.resp_high, runtime.step_size)

    with timings.stage("subsample"):
        X, y = build_sample(X, y, config.data.n_samples, rng)

    with timings.stage("environment"):
        env = collect()

    json_dump(run_dir, config, env, timings)

def add_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--top-k", type=int, default=DataConfig.top_k, help="Top k (default: %(default)s)")
    parser.add_argument("--resp-low", type=float, default=DataConfig.resp_low, help="Response low (default: %(default)s)")
    parser.add_argument("--resp-high", type=float, default=DataConfig.resp_high, help="Response high (default: %(default)s)")
    parser.add_argument("--n-samples", type=int, default=DataConfig.n_samples, help="Number of samples (default: %(default)s)")
    parser.add_argument("--path", type=str, default=DataConfig.path, help="Path (default: %(default)s)")
    parser.add_argument("--tree-name", type=str, default=DataConfig.tree_name, help="Tree name (default: %(default)s)")

    return parser

def add_runtime_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=RunConfig.seed, help="Random seed (default: %(default)s)")
    parser.add_argument("--log-level", type=str, default=RuntimeConfig.log_level, choices=sorted(LOG_LEVELS), help="Logging level (default: %(default)s)")
    parser.add_argument("--step-size", type=int, default=RuntimeConfig.step_size, help="Step size (default: %(default)s)")
    parser.add_argument("--runs-dir", type=str, default=RuntimeConfig.runs_dir, help="Directory for runs (default: %(default)s)")

    return parser

def create_run_dir(runtime: RuntimeConfig, config: Config) -> Path:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runtime.runs_dir) / f"{now}-{config.config_hash()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def create_rng(config: Config, log: logging.Logger) -> Generator:
    rng = set_seed(config.run.seed)
    log.info("Running with seed %s", config.run.seed)
    return rng

def json_dump(run_dir: Path, config: Config, env: dict, timings: Timings) -> None:
    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    with (run_dir / "env.json").open("w") as f:
        json.dump(env, f, indent=2)
    
    with (run_dir / "timings.json").open("w") as f:
        json.dump(timings.as_dict(), f, indent=2)