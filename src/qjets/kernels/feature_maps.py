from qiskit.circuit.library import ZZFeatureMap

def build_feature_map(n_qubits: int, reps: int, entanglement: str) -> ZZFeatureMap:
    return ZZFeatureMap(n_qubits, reps=reps, entanglement=entanglement)