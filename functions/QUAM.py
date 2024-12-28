#------------------------------------------------------------------------------
# quam_utilities.py
#
# This module provides utilities for implementing quantum memory and search
# algorithms, including multi-qubit gates, Grover's search, and Quantum Associative Memory (QuAM).
#
# The module includes the following functions:
# - multi_phase_gate(qc, q, theta): Applies multi-qubit controlled phase rotation using a specified angle.
# - multi_CX_gate(qc, q_controls, q_target, sig): Applies a multi-qubit controlled X gate based on control qubits and signature.
# - multi_CZ_gate(qc, q_controls, q_target, sig): Applies a multi-qubit controlled Z gate based on control qubits and signature.
# - get_state(qc, x): Retrieves the state of the quantum register and returns it as a string.
# - flip(qc, x, c, patterns, index): Applies controlled-X gates based on a change in bit pattern.
# - S_p(qc, c, p): Applies a controlled-U gate based on the value of a parameter p.
# - save(qc, x, c, patterns): Saves a list of bit string patterns into quantum memory.
# - update(x, q): Parses a query string and updates the quantum register accordingly.
# - grover_diffusion(qc, x): Performs the Grover diffusion operation (inversion about the mean).
# - grover_search(qc, x, c, output, xc, cc, R, s, patterns, problem): Implements the Grover search algorithm.
# - QuAM(patterns, search=None): Implements Quantum Associative Memory (QuAM) to search for a target bit string among patterns.
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit_algorithms import AmplificationProblem
from qiskit_algorithms import Grover
from qiskit.primitives import Sampler
from typing import List, Tuple
import numpy as np
from functions.patterns_utilities import generate_expression


def get_pattern(n: int, s) -> str:
    """
    Get a bit string pattern.

    Args:
        n (int): pattern length corresponding to 2^n bits.
        s (int or str): bit string pattern. If s is str returns it as it is. If s is int converts it to
                        a binary string then retruns it.
    Returns:
        str: bit string pattern.
    """
    if type(s) is int:
        assert s >= 0 and s < 2 ** n, "Invalid bit string, it has the wrong length."
        return ("{0:0" + str(n) + "b}").format(s)
    elif type(s) is str:
        assert len(s) == n
        return s

def multi_phase_gate(qc: QuantumCircuit, q: QuantumRegister, theta: float):
    """
    Multi-qubit controlled phase rotation. Applies a phase factor exp(i*theta) if all the qubits are 1.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        q (QuantumRegister): Quantum Register in qc.
        theta (float): Rotation angle.
        verbose (bool): If True enters in debugging mode. The default is False.

    Note: It doesn't matter which qubits are the controls and which is the target.
    """
    #q = [q[i] for i in range(n)]
    if len(q) == 1:
        qc.u1(theta, q[0])
    elif len(q) == 2:
        qc.cp(theta, q[0], q[1])
    else:
        qc.cp(theta / 2, q[1], q[0])
        multi_CX_gate(qc, q[2:], q[1])
        qc.cp(-theta / 2, q[1], q[0])
        multi_CX_gate(qc, q[2:], q[1])
        multi_phase_gate(qc, [q[0]] + q[2:], theta / 2)


def multi_CX_gate(qc: QuantumCircuit, q_controls: QuantumRegister, q_target: QuantumRegister,
                  sig: list = None):
    """
    Multi-qubit controlled X gate. Applies an X gate to q_target if q_controls[i] == sig[i] for all i.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        q_controls (QuantumRegister): Quantum Register of the controlling qubits.
        q_target (QuantumRegister): Quantum Register of the target qubit.
        sig (list): Signature of the control. The default is set to [1,1,...,1].

    Note: Since X = H*Z*H we can write CX = H*CZ*H.
    """
    if sig is None:
        sig = [1] * len(q_controls)
    qc.h(q_target)
    multi_CZ_gate(qc, q_controls, q_target, sig)
    qc.h(q_target)


def multi_CZ_gate(qc: QuantumCircuit, q_controls: QuantumRegister, q_target: QuantumRegister,
                  sig: str = None):
    """
    Multi-qubit controlled Z gate. Applies a Z gate to q_target if q_controls[i] == sig[i] for all i.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        q_controls (QuantumRegister): Quantum Register of the controlling qubits.
        q_target (QuantumRegister): Quantum Register of the target qubit.
        sig (str): Signature of the control. The default is set to "11...1".
    """
    if sig is None:
        sig = "1" * len(q_controls)
    #apply signature gates
    for i in range(len(q_controls)):
        if sig[i] == "0":
            qc.x(q_controls[i])
    q = [q_controls[i] for i in range(len(q_controls))] + [q_target]
    multi_phase_gate(qc, q, np.pi)
    #undo signature gates
    for i in range(len(q_controls)):
        if sig[i] == "0":
            qc.x(q_controls[i])


def get_state(qc: QuantumCircuit, x: QuantumRegister) -> str:
    """
    Get the state of the quantum register x.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        x (QuantumRegister): Quantum Register.

    Returns:
        str: State of the quantum register.
    """
    n = len(x)
    result = ""
    probs = Statevector.from_instruction(qc).probabilities()
    for state in range(len(probs)):
        extra_bits = 3
        bformat = "{0:0" + str(n + extra_bits) + "b}"
        state_str = bformat.format(state)
        state_str = state_str[extra_bits:]
        if np.abs(probs[state]) < 0.0001:
            continue
        elif probs[state] == 1:
            return "|" + state_str + ">"
        coefficient = np.sqrt(probs[state])
        coefficient = "{0:.2f}".format(coefficient)
        result += coefficient + " |" + state_str + "> + "
    result = result[: len(result) - 3]

    return result


def flip(qc: QuantumCircuit, x: QuantumRegister, c: ClassicalRegister, patterns: List[str], index: int):
    """
    Flip gate. Applies controlled X gates based on the change of bit pattern.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        x (QuantumRegister): Quantum Register for input bits.
        c (ClassicalRegister): Classical Register for output bits.
        patterns (list): List of bit string patterns.
        index (int): Index of the current pattern.
    """
    n = len(x)
    prev_pattern = ""
    if index == 0:
        prev_pattern = "0" * n
    else:
        prev_pattern = patterns[index - 1]
    pattern = patterns[index]
    # pattern inversion to match Qiskit qubit conventions
    pattern = pattern[::-1]
    prev_pattern = prev_pattern[::-1]
    for i in range(n):
        if pattern[i] != prev_pattern[i]:
            multi_CX_gate(qc, c, x[i], "00")
    multi_CX_gate(qc, x, c[1], pattern)


def S_p(qc: QuantumCircuit, c: ClassicalRegister, p: int):
    """
    S_p gate. Applies controlled-U gate based on the value of p.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        c (ClassicalRegister): Classical Register for output bits.
        p (int): Parameter.
    """
    theta = 2 * np.arccos(np.sqrt((p - 1) / p))
    qc.cu(theta, 0, 0, 0, c[1], c[0])


def save(qc: QuantumCircuit, x: QuantumRegister, c: ClassicalRegister, patterns: List[str]):
    """
    Save multiple bit string patterns.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        x (QuantumRegister): Quantum Register for input bits.
        c (ClassicalRegister): Classical Register for output bits.
        output (List): List of output bits.
        xc (QuantumRegister): Ancilla Quantum Register.
        cc (ClassicalRegister): Ancilla Classical Register.
        patterns (List[int]): List of bit string patterns to be saved.
    """
    n = len(x)
    m = len(patterns)
    assert m <= 2 ** n
    for i in range(m):
        pattern = get_pattern(n, patterns[i])
        assert len(pattern) == n
        flip(qc, x, c, patterns, i)
        S_p(qc, c, m - i)
        multi_CX_gate(qc, x, c[1], pattern[::-1])


def update(x: QuantumRegister, q: str) -> Tuple[QuantumRegister, str]:
    """
    Parse the query string and update the quantum register accordingly.

    Args:
        x (QuantumRegister): Quantum Register for input bits.
        q (str): Query string.

    Returns:
        tuple: Updated quantum register and query string.
    """
    x_register = x
    i = 0
    while i < len(q):
        if q[i] == "?":
            x_register = x_register[:i] + x_register[i + 1 :]
            q = q[:i] + q[i + 1 :]
        else:
            i += 1

    return x_register, q

def grover_diffusion(qc: QuantumCircuit, x: QuantumRegister):
    """
    Grover Diffusion operation. Performs the inversion about the mean operation in Grover's algorithm.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        x (QuantumRegister): Quantum Register for input bits.
    """
    qc.h(x)
    qc.x(x)
    multi_phase_gate(qc, x, np.pi)
    qc.x(x)
    qc.h(x)


def grover_search(qc: QuantumCircuit, x: QuantumRegister, c: ClassicalRegister,
                 output: list, xc: QuantumRegister, cc: ClassicalRegister,
                 R: int, s: int, patterns: List[str], problem:"str"="pattern_completion"):
    """
    Implements the Grover search algorithm with a modified grover iterate.

    Args:
        qc (QuantumCircuit): Quantum Circuit.
        x (QuantumRegister): Quantum Register for input bits.
        c (ClassicalRegister): Classical Register for output bits.
        output (list): List of output bits.
        xc (QuantumRegister): Ancilla Quantum Register.
        cc (ClassicalRegister): Ancilla Classical Register.
        R (int): Number of iterations.
        s (int): Target bit pattern.
        ....

    Note: The "vanilla" Grover search algorithm requires an equal superposition as a starting point. Since
        we do not have such an initial state, we slightly modify the second Grover iteration. Rather then
        applying the unitary rotating the target state we apply a phase rotation to every state in the
        initial superposition. This is achieved by controlling the x register, identifying each initial
        saved pattern.
    """
    if problem == "pattern_completion":
        n = len(x)
        s = get_pattern(n, s)[::-1]
        qc.x(output[0])
        qc.h(output[0])
        x_register, s = update(x, s)
        multi_CX_gate(qc, x_register, output, s)
        #first (unaltered) iteration
        grover_diffusion(qc, x)
        #modified Grover iteration (initial state not in uniform superposition)
        for pattern in patterns:
            multi_CX_gate(qc, x, output, get_pattern(n, pattern)[::-1])
        grover_diffusion(qc, x)
        #remaining (unaltered) iterations
        for i in range(R - 2):
            multi_CX_gate(qc, x_register, output, s)
            grover_diffusion(qc, x)
        qc.h(output[0])
        qc.x(output[0])
        qc.measure(x, xc)
        qc.measure(c, cc)
    elif problem == "similarity":
        expression = generate_expression(patterns=patterns, search=s)
        oracle = Statevector.from_label(expression)
        problem = AmplificationProblem(oracle, is_good_state=oracle)
        grover = Grover(sampler=Sampler())
        return grover.amplify(problem)


def QuAM(patterns: list, search: str = None):
    """
    Quantum Associative Memory (QuAM). Can search for a target bit string 'search'
    among the given bit string patterns.

    Args:
        patterns (list): List of bit string patterns.
        search (str): Target bit string to search for. The default is None

    Returns:
        tuple: Quantum state and data.

    Note: if search is None the QuAM stores the pattenrs into a quantum state without solving
          the associative pattern completion problem.
    """
    n = len(patterns[0])
    N = 2 ** n
    R = int(np.floor(np.pi * np.sqrt(N) / 4))
    x = QuantumRegister(n)
    c = QuantumRegister(2)
    xc = ClassicalRegister(n)
    cc = ClassicalRegister(2)
    output = QuantumRegister(1)
    qc = QuantumCircuit(x, c, output, xc, cc)
    save(qc, x, c, patterns)
    state = get_state(qc, x)
    if search is not None:
        grover_search(qc, x, c, output, xc, cc, R, search, patterns)
        backend = Aer.get_backend("qasm_simulator")
        t_qc = transpile(qc, backend=backend)
        job = backend.run(t_qc)
        data = job.result().get_counts(qc)
        return state, data
    elif search is None:
        return state, None
