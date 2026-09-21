import os
import platform
from importlib.metadata import version, PackageNotFoundError

PACKAGES = (
    "numpy",
    "scikit-learn",
    "matplotlib",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
    "uproot",
    "awkward",)

def environment() -> dict[str, object]:
    return {
        "os": platform.platform(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "num_cpus": os.cpu_count(),
    }

def package_versions() -> dict[str, str | None]:
    result = {}
    for name in PACKAGES:
        result[name] = package_version(name)

    return result

def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def available_devices() -> list[str] | None:
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator().available_devices()
    except Exception:
        return None

def collect() -> dict[str, object]:
    return {
        "environment": environment(),
        "package_versions": package_versions(),
        "available_devices": available_devices(),
    }