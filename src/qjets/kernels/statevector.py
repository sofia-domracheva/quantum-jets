import logging
import numpy as np
from sklearn.svm import SVC
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.circuit import QuantumCircuit
from qjets.kernels.feature_maps import build_feature_map
from qjets.models.classical import evaluate
from qjets.config import QuantumConfig


log = logging.getLogger(__name__)

def make_simulator(device: str) -> AerSimulator:
    available_devices = AerSimulator().available_devices()
    if device == "auto":
        if "CPU" in available_devices:
            device = "CPU"
        else:
            device = "GPU"
    elif device == "GPU":
        if "GPU" not in available_devices:
            log.warning("GPU device not available, falling back to CPU")
            device = "CPU"
    return AerSimulator(method="statevector", device=device)

def statevectors(feature_map: QuantumCircuit, X: np.ndarray, simulator: AerSimulator, batch_size: int) -> np.ndarray:
    quantum_circuit = feature_map.copy()
    quantum_circuit.save_statevector()
    quantum_circuit = transpile(quantum_circuit, simulator)

    result = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_schemes = [quantum_circuit.assign_parameters(x) for x in batch]
        state = simulator.run(batch_schemes).result()
        for k in range(len(batch_schemes)):
            result.append(np.array(state.get_statevector(k)))
    return np.array(result)

def kernel_matrix(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    return np.abs(np.dot(V1.conj(), V2.T))**2

def run_quantum(datasets: dict[int, dict[str, np.ndarray]], y_train: np.ndarray, y_test: np.ndarray, device: str, batch_size: int, config: QuantumConfig, seed: int) -> dict[int, dict]:
    results = {}
    simulator = make_simulator(device)
    for n, data in datasets.items():
        feature_map = build_feature_map(n, config.reps, config.entanglement)

        test_state = statevectors(feature_map, data["test"], simulator, batch_size)
        train_state = statevectors(feature_map, data["train"], simulator, batch_size)

        test_matrix = kernel_matrix(test_state, train_state)
        train_matrix = kernel_matrix(train_state, train_state)

        svc_model = SVC(kernel="precomputed", C=1.0).fit(train_matrix, y_train)
        metrics = evaluate(svc_model, test_matrix, y_test)
        m = len(train_matrix)
        concentration = (train_matrix.sum()-np.trace(train_matrix)) / (m*(m-1))
        results[n] = {"metrics": metrics, "concentration":concentration, "theory": 2**(-n)}
        log.info(f"Results for {n} qubits: {metrics}, concentration: {concentration}, theory: {2**(-n)}")

    return results