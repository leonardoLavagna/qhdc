#------------------------------------------------------------------------------
# circuits_utilities.py
#
# This module provides functions for generating various quantum circuits.
# The circuits include random circuits with a high probability of the zero state,
# tensor product circuits, entangled circuits, and circuits with general rotations.
#
# The module includes the following functions:
# - create_random_circuit_with_high_zero_prob(n): Generates a random quantum circuit
#   with n qubits that has a high probability of measuring the zero state.
# - tensor_product_circuits(qc_1, qc_2): Creates a new quantum circuit that is the
#   tensor product of two given quantum circuits.
# - create_entangled_circuit(qc_1, qc_2, entanglement_pattern): Creates a new quantum circuit
#   combining two circuits with entanglement specified by a pattern.
# - create_rotation_circuit(n, theta): Creates a quantum circuit applying a general rotation
#   to every qubit, using the specified angle.
# - generate_random_circuit_from_sequence(sequence,n,depth): Generate a random quantum circuit
#   with high probability of measuring the zero state given a (DNA) sequence.
# - measure_zero_probability(qc): Execute a circuit and calculate the probability of measuring 
#   the zero state
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
import random
import numpy as np


def create_random_circuit_with_high_zero_prob(n: int) -> QuantumCircuit:
    """
    Create a random quantum circuit with n qubits and n classical bits that has
    a high probability of resulting in the zero state when measured.

    Args:
        n (int): Number of qubits and classical bits in the circuit.

    Returns:
        QuantumCircuit: A quantum circuit that has a high probability of measuring the zero state.
    """
    qc = QuantumCircuit(n, n)
    for qubit in range(n):
        qc.h(qubit)
    for _ in range(n):
        control_qubit = random.randint(0, n-1)
        target_qubit = random.randint(0, n-1)
        if control_qubit != target_qubit:
            qc.cx(control_qubit, target_qubit)
    for qubit in range(n):
        qc.h(qubit)

    return qc


def tensor_product_circuits(qc_1: QuantumCircuit, qc_2: QuantumCircuit) -> QuantumCircuit:
    """
    Create a new quantum circuit that is the tensor product of qc_1 and qc_2.

    Args:
        qc_1 (QuantumCircuit): The first quantum circuit.
        qc_2 (QuantumCircuit): The second quantum circuit.

    Returns:
        QuantumCircuit: A quantum circuit that is the tensor product of qc_1 and qc_2.
    """
    n_1 = qc_1.num_qubits
    n_2 = qc_2.num_qubits
    qr = QuantumRegister(n_1 + n_2)
    cr = ClassicalRegister(n_1 + n_2)
    qc = QuantumCircuit(qr, cr)
    qc.append(qc_1, qr[:n_1])
    qc.append(qc_2, qr[n_1:])

    return qc

def juxtaposition_circuits(qc_1: QuantumCircuit, qc_2: QuantumCircuit) -> QuantumCircuit:
    """
    Create a new quantum circuit that is the juxtaposition of qc_1 and qc_2.

    Args:
        qc_1 (QuantumCircuit): The first quantum circuit.
        qc_2 (QuantumCircuit): The second quantum circuit.

    Returns:
        QuantumCircuit: A quantum circuit that is the juxtaposition of qc_1 and qc_2.
    """
    assert qc_1.num_qubits == qc_2.num_qubits, "the circuits must have the same number of qubits"
    return qc_1.compose(qc_2)


def create_entangled_circuit(qc_1: QuantumCircuit, qc_2: QuantumCircuit,
                             entanglement_pattern: list) -> QuantumCircuit:
    """
    Create a new quantum circuit that combines qc_1 and qc_2 with entanglement.

    Args:
        qc_1 (QuantumCircuit): The first quantum circuit.
        qc_2 (QuantumCircuit): The second quantum circuit.
        entanglement_pattern (list): A list of tuples representing the entanglement pairs.
                                      Each tuple is of the form (qubit_in_qc1, qubit_in_qc2).

    Returns:
        QuantumCircuit: A quantum circuit that is a combination of qc_1 and qc_2 with entanglement.
    """
    n = qc_1.num_qubits
    m = qc_2.num_qubits
    if n != m:
        raise ValueError("Both circuits must have the same number of qubits.")
    qr = QuantumRegister(2 * n)
    cr = ClassicalRegister(2 * n)
    qc = QuantumCircuit(qr, cr)
    qc.append(qc_1, qr[:n])
    qc.append(qc_2, qr[n:])
    for (i, j) in entanglement_pattern:
        qc.cx(qr[i], qr[n + j])

    return qc


def create_rotation_circuit(n: int, theta: float) -> QuantumCircuit:
    """
    Create a quantum circuit that applies a general rotation to every qubit.

    Args:
        n (int): Number of qubits in the circuit.
        theta (float): Angle of rotation in radians to be applied to each qubit.

    Returns:
        QuantumCircuit: A quantum circuit with the specified rotations applied to each qubit.
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    for qubit in range(n):
        qc.rx(theta, qr[qubit])
        qc.ry(theta, qr[qubit])
        qc.rz(theta, qr[qubit])

    return qc

def generate_random_circuit_from_sequence(sequence, n=3, depth=5):
    """
    Generate a random quantum circuit with high probability of measuring |0> given a (DNA) sequence.

    Args: 
        sequence (List[char]): A list of (DNA) strings.
        n (int): Number of qubits (default 3).
        depth (int): Depth (default 5).
    Returns:
        QuantumCircuit: A quantum circuit that has a high probability of measuring the zero state.
    """
    qc = QuantumCircuit(n)
    # Apply Hadamard to introduce superposition
    for qubit in range(n):
        qc.h(qubit)
    # Random gates with bias towards |0>
    for _ in range(depth):
        for qubit in range(n):
            if np.random.rand() < 0.7: 
                qc.rz(np.random.uniform(0, np.pi / 4), qubit)
            else:
                qc.rx(np.random.uniform(0, np.pi), qubit)
    
    qc.measure_all()
    return qc

def measure_zero_probability(qc):
    """
    Execute a circuit and calculate the probability of measuring the zero state.

    Args: 
        qc (QuantumCircuit): A quantum circuit.

    Returns:
        float: the probability of measuring the zero state
    """
    simulator = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend=simulator)
    job = simulator.run(t_qc)
    result = job.result()
    counts = result.get_counts()
    p_zero = counts.get('0' * qc.num_qubits, 0) / 1024
    return p_zero
