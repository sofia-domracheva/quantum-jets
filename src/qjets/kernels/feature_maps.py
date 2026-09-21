from qiskit.circuit.library import zz_feature_map
from qiskit.circuit import QuantumCircuit

def build_feature_map(n_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)