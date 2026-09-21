import argparse
import json
import logging

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from numpy.random import Generator
import qjets.config as qjc
from qjets.runtime.log import setup_logging
from qjets.runtime.seeding import set_seed
from qjets.runtime.timing import Timings
from qjets.runtime.environment import collect
from qjets.data.features import build_sample
from qjets.data.root_io import load_jets_root
from qjets.preprocessing import prepare_all
from qjets.models.classical import run_baseline
from qjets.kernels.statevector import run_quantum

def main() -> None:
    parser = argparse.ArgumentParser(description="Classical vs quantum SVM on LHC jet data", prog="qjets")
    parser = add_data_arguments(parser)
    parser = add_runtime_arguments(parser)
    parser = add_preprocessing_arguments(parser)
    parser = add_model_arguments(parser)
    parser = add_quantum_arguments(parser)

    args = parser.parse_args()

    runtime = qjc.RuntimeConfig(
        log_level=args.log_level, 
        step_size=args.step_size, 
        runs_dir=args.runs_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    data = qjc.DataConfig(
        top_k=args.top_k, 
        resp_low=args.resp_low, 
        resp_high=args.resp_high, 
        n_samples=args.n_samples, 
        path=args.path, 
        tree_name=args.tree_name)

    prep = qjc.PreprocessingConfig(
        test_size=args.test_size,
        qubit_dims=tuple(args.qubit_dims)
    )

    model = qjc.ModelConfig(
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        kernel=args.kernel
    )

    quantum = qjc.QuantumConfig(
        reps=args.reps,
        entanglement=args.entanglement,
    )
    
    config = qjc.Config(run=qjc.RunConfig(seed=args.seed), data=data, prep=prep, model=model, quantum=quantum)

    run_dir = create_run_dir(runtime, config)
    log = setup_logging(runtime.log_level, run_dir)
    rng = create_rng(config, log)

    timings = Timings()

    with timings.stage("load"):
        X, y = load_jets_root(
            Path(config.data.path), 
            config.data.tree_name, 
            config.data.top_k, 
            config.data.resp_low, 
            config.data.resp_high, 
            runtime.step_size)

    with timings.stage("subsample"):
        X, y = build_sample(X, y, config.data.n_samples, rng)

    with timings.stage("preprocess"):
        datasets, y_train, y_test = prepare_all(X, y, config.prep, config.run.seed)

    with timings.stage("classical"):
        classical_results = run_baseline(datasets, y_train, y_test, config.model, config.run.seed)

    with timings.stage("quantum"):
        quantum_results = run_quantum(datasets, y_train, y_test, runtime.device, runtime.batch_size, config.quantum, config.run.seed)

    with timings.stage("environment"):
        env = collect()

    results = {
        "classical": classical_results,
        "quantum": quantum_results
    }
    json_dump(run_dir, config, env, timings, results)

def add_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--top-k", type=int, default=qjc.DataConfig.top_k, help="Top k (default: %(default)s)")
    parser.add_argument("--resp-low", type=float, default=qjc.DataConfig.resp_low, help="Response low (default: %(default)s)")
    parser.add_argument("--resp-high", type=float, default=qjc.DataConfig.resp_high, help="Response high (default: %(default)s)")
    parser.add_argument("--n-samples", type=int, default=qjc.DataConfig.n_samples, help="Number of samples (default: %(default)s)")
    parser.add_argument("--path", type=str, default=qjc.DataConfig.path, help="Path (default: %(default)s)")
    parser.add_argument("--tree-name", type=str, default=qjc.DataConfig.tree_name, help="Tree name (default: %(default)s)")
    return parser

def add_runtime_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=qjc.RunConfig.seed, help="Random seed (default: %(default)s)")
    parser.add_argument("--log-level", type=str, default=qjc.RuntimeConfig.log_level, choices=sorted(qjc.LOG_LEVELS), help="Logging level (default: %(default)s)")
    parser.add_argument("--step-size", type=int, default=qjc.RuntimeConfig.step_size, help="Step size (default: %(default)s)")
    parser.add_argument("--runs-dir", type=str, default=qjc.RuntimeConfig.runs_dir, help="Directory for runs (default: %(default)s)")
    parser.add_argument("--device", type=str, default=qjc.RuntimeConfig.device, choices=sorted(qjc.DEVICES), help="Device (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=qjc.RuntimeConfig.batch_size, help="Batch size (default: %(default)s)")
    return parser

def add_preprocessing_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--test-size", type=float, default=qjc.PreprocessingConfig.test_size, help="Test size (default: %(default)s)")
    parser.add_argument("--qubit-dims", type=int, nargs="+", default=qjc.PreprocessingConfig.qubit_dims, help="Qubit dimensions (default: %(default)s)")
    return parser

def add_model_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--cv-folds", type=int, default=qjc.ModelConfig.cv_folds, help="Number of CV folds (default: %(default)s)")
    parser.add_argument("--scoring", type=str, default=qjc.ModelConfig.scoring, help="Scoring metric (default: %(default)s)")
    parser.add_argument("--kernel", type=str, default=qjc.ModelConfig.kernel, help="Kernel (default: %(default)s)")
    return parser

def add_quantum_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--reps", type=int, default=qjc.QuantumConfig.reps, help="Number of reps (default: %(default)s)")
    parser.add_argument("--entanglement", type=str, default=qjc.QuantumConfig.entanglement, help="Entanglement (default: %(default)s)")
    return parser

def create_run_dir(runtime: qjc.RuntimeConfig, config: qjc.Config) -> Path:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runtime.runs_dir) / f"{now}-{config.config_hash()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def create_rng(config: qjc.Config, log: logging.Logger) -> Generator:
    rng = set_seed(config.run.seed)
    log.info("Running with seed %s", config.run.seed)
    return rng

def json_dump(run_dir: Path, config: qjc.Config, env: dict, timings: Timings, results: dict) -> None:
    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    with (run_dir / "env.json").open("w") as f:
        json.dump(env, f, indent=2)
    
    with (run_dir / "timings.json").open("w") as f:
        json.dump(timings.as_dict(), f, indent=2)

    with (run_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=float)