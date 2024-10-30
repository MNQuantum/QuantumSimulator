"""
Quantum computing simulator using NumPy tensors.

    Author: David Radcliffe (dradcliffe@gmail.com)
    URL: https://github.com/MNQuantum/QuantumSimulator
    License: GPLv2. See LICENSE file for more information.
    Last updated: 29 October 2024

An n-qubit quantum register is represented by a tensor of complex numbers
of shape (2, 2, ..., 2), containing 2^n complex numbers. Each complex number
requires 128 bits of memory (using np.complex128). The sum of the squared
magnitudes of the amplitudes in a quantum register must equal one.

Quantum gates are represented as unitary complex tensors.
  *  A unary gate is a tensor of shape (2, 2).
  *  A binary gate is a tensor of shape (2, 2, 2, 2).
  *  A ternary gate is a tensor of shape (2, 2, 2, 2, 2, 2).
Quantum gates are applied using tensor contraction, which generalizes matrix
multiplication to higher dimensions.

While the simulator supports gates acting on any number of qubits, any quantum
circuit can be expressed using only one- and two-qubit gates. Three-qubit gates
are rarely used in practice as they can be decomposed into simpler gates and
are expensive to implement physically.

The quantum gate functions use a higher-order functional design: each gate
function returns another function that can be applied to quantum registers.
This enables easy composition of gates into circuits.

Measurement follows the quantum measurement postulates:
  * Measuring n qubits returns a binary vector of length n
  * The probability of measuring a particular state is given by the squared
    magnitude of its amplitude (Born rule)
  * Measurement causes wavefunction collapse: the register's state changes
    to reflect the measurement outcome
  * Both full and partial measurements are supported

If you use this code in your work, please credit the author and include a
link to the repository.
"""

from itertools import product
import numpy as np
from collections import Counter
from typing import Callable, List, Tuple


def qubit() -> np.ndarray:
    """Create a qubit, initialized to |0〉.
    Returns a complex (np.complex128) tensor of shape (2,).
    Equivalent to quantum_register(1).
    
    Returns:
        np.ndarray: A complex tensor of shape (2,) representing the qubit.
    """
    return np.array([1, 0j], dtype=np.complex128)


def random_qubit() -> np.ndarray:
    """Create a qubit with random complex amplitudes. Equivalent to random_register(1).

    Returns:
        np.ndarray: A complex tensor of shape (2,) representing the qubit.
    Returns a complex (np.complex128) tensor of shape (2,).
    """
    q = np.random.standard_normal(2) + 1j * np.random.standard_normal(2)
    q /= np.linalg.norm(q)
    return q


def quantum_register(number_of_qubits: int) -> np.ndarray:
    """Creates a linear register of qubits, initialized to |000...0〉.

    Args:
        number_of_qubits (int): The number of qubits in the register.

    Returns:
        np.ndarray: A complex tensor of shape (2, 2, ..., 2) representing the quantum register.
    """
    shape = (2,) * number_of_qubits
    first = (0,) * number_of_qubits
    register = np.zeros(shape, dtype=np.complex128)
    register[first] = 1+0j
    return register


def random_register(number_of_qubits: int) -> np.ndarray:
    """Creates a linear register of qubits with random complex amplitudes.
    Returns a complex (np.complex128) tensor of shape (2, 2, ..., 2).
    
    Args:
        number_of_qubits (int): The number of qubits in the register.
    
    Returns:
        np.ndarray: A complex tensor representing the quantum register.
    """
    shape = (2,) * number_of_qubits
    register = np.random.standard_normal(shape) + np.random.standard_normal(shape) * 1j
    register = register / np.linalg.norm(register)
    return register


def display(tensor: np.ndarray) -> None:
    """Displays the tensor entries in tabular form, showing quantum basis states
    and their complex amplitudes.

    Args:
        tensor (np.ndarray): The quantum state tensor to display.

    Returns:
        None

    Example:
        >>> state = (np.array([[1, 2], [3, 4]], dtype=np.complex128) / np.sqrt(30))
        >>> display(state)
        |00⟩    0.18257 + 0.00000 i
        |01⟩    0.36515 + 0.00000 i
        |10⟩    0.54772 + 0.00000 i
        |11⟩    0.73030 + 0.00000 i

    Note:
        The function ensures proper handling of both scalar values and
        0-dimensional arrays that might result from tensor indexing.
    """
    for multiindex in product(*map(range, tensor.shape)):
        # Create the ket label (e.g., |00⟩, |01⟩, etc.)
        ket = '|' + ''.join(str(i) for i in multiindex) + '⟩'

        # Get the value and ensure it's a scalar
        value = tensor[multiindex]
        if isinstance(value, np.ndarray):
            value = value.item()  # Convert 0-dim array to scalar

        # Format and print the result
        print(f'{ket:<8}{value.real:>8.5f} + {value.imag:>8.5f} i')


def get_transposition(n: int, indices: Tuple[int, ...]) -> List[int]:
    """Helper function that reorders a tensor after a quantum gate is applied.
    
    This function generates a permutation that restores the correct ordering of 
    tensor dimensions after a quantum gate operation. The resulting permutation
    ensures that the operated qubits are moved to their original positions.
    
    Args:
        n (int): Total number of qubits in the register (tensor rank)
        indices (Tuple[int, ...]): Indices of qubits that the gate operates on
    
    Returns:
        List[int]: Permutation that restores correct tensor dimension ordering
    
    Example:
        For n=3 qubits and indices=[1], applying a single-qubit gate to the second qubit:
        get_transposition(3, [1]) returns [0, 2, 1]
        This moves the operated qubit from position 2 back to position 1
    """
    transpose = [0] * n      # Initialize permutation list
    k = len(indices)         # Number of qubits the gate operates on
    ptr = 0                  # Pointer for non-operated qubits
    for i in range(n):
        if i in indices:
            # Operated qubits get moved from the end (where tensordot puts them)
            # back to their original positions
            transpose[i] = n - k + indices.index(i)
        else:
            # Non-operated qubits are placed sequentially at the start
            transpose[i] =  ptr
            ptr += 1
    return transpose


def apply_gate(gate: np.ndarray, *indices: int) -> Callable:
    """Applies a gate to one or more indices of a quantum register.
    This is a higher-order function: it returns another function that
    can be applied to a register."""
    axes = (indices, range(len(indices)))
    def op(register):
        return np.tensordot(register, gate, axes=axes).transpose(
               get_transposition(register.ndim, indices))
    return op


def circuit(ops: List[Callable]) -> Callable:
    """Constructs a circuit as a sequence of quantum gates.
    This higher-order function returns another function that
    can be applied to a quantum register."""
    def circ(register):
        for op in ops:
            register = op(register)
        return register
    return circ


def measure_circuit(ops: List[Callable], index=None) -> Callable:
    """Constructs a circuit and performs a measurement."""
    circ = circuit(ops)
    def m(register):
        if index is None:
            return measure_all() (circ(register))
        return measure(index) (circ(register))
    return m

##### Unary gates

def X(index: int) -> Callable:
    """Generates a Pauli X gate (also called a NOT gate) acting on a given index.
    It returns a function that can be applied to a register."""
    gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    return apply_gate(gate, index)


def Y(index: int) -> Callable:
    """Generates a Pauli Y gate acting on a given index.
    It returns a function that can be applied to a register."""
    gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    return apply_gate(gate, index)


def Z(index: int) -> Callable:
    """Generates a Pauli Z gate acting on a given index.
    This is the same as a rotation of the Bloch sphere by pi radians about the Z-axis.
    It returns a function that can be applied to a register."""
    gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return apply_gate(gate, index)


def R(index: int, angle: float) -> Callable:
    """Generates a rotation of the Block sphere about the Z-axis by a given
    It returns a function that can be applied to a register."""
    gate = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=np.complex128)
    return apply_gate(gate, index)


def S(index: int) -> Callable:
    """The S gate is a 90-degree rotation of the Bloch sphere about the Z-axis."""
    return R(index, np.pi/2)


def T(index: int) -> Callable:
    """The T gate is a 45-degree rotation of the Bloch sphere about the Z-axis."""
    return R(index, np.pi/4)


def H(index: int) -> Callable:
    """Generates a Hadamard gate. It returns a function that can be applied to a register."""
    gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.)
    return apply_gate(gate, index)


def SNOT(index: int) -> Callable:
    """Generates a 'square root of NOT' gate. It returns a function that can be applied to a register."""
    gate = np.array([[1+1j, 1-1j], [1-1j, 1+ 1j]], dtype=np.complex128) / 2.
    return apply_gate(gate, index)


### Binary gates


def SWAP(i: int, j: int) -> Callable:
    """Generates a SWAP gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 0, 1, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1], dtype=np.complex128
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def CNOT(i: int, j: int) -> Callable:
    """Generates a controlled NOT gate, also called a controlled X gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 1, 0], dtype=np.complex128
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate,i, j)

CX = CNOT


def CY(i: int, j: int) -> Callable:
    """Generates a controlled Y gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, -1j,
                     0, 0, 1j, 0], dtype=np.complex128
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def CZ(i: int, j: int) -> Callable:
    """Generates a controlled Z gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, -1], dtype=np.complex128
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def C(i: int, j: int, unary_gate: np.ndarray) -> Callable:
    """Generates a controlled (binary) version of a unary gate.
    When the value at the first index is 1, apply the unary gate to the value at the second index.
    When the value at the first index is 0, do nothing."""
    gate = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    gate[0, :, 0, :] = np.eye(2)
    gate[1, :, 1, :] = unary_gate
    return apply_gate(gate, i, j)


## Ternary gates

def CCNOT(i: int, j: int, k: int) -> Callable:
    """Generates a Toffoli gate."""
    gate = np.eye(8, dtype=np.complex128)
    gate[6:8, 6:8] = np.array([[0, 1], [1, 0]])
    gate = gate.reshape((2, 2, 2, 2, 2, 2))
    return apply_gate(gate, i, j, k)


def CSWAP(i: int, j: int, k: int) -> Callable:
    """Generates a Fredkin gate, or a controlled SWAP gate."""
    gate = np.eye(8, dtype=np.complex128)
    gate[5:7, 5:7] = np.array([[0, 1], [1, 0]])
    gate = gate.reshape((2, 2, 2, 2, 2, 2))
    return apply_gate(gate, i, j, k)

## Quantum circuits

def bell_state(i: int, j: int) -> Callable:
    """Generates an entangled Bell state on two qubits."""
    def b(register):
        return CNOT(i, j) (H(i) (register))
    return b


## MEASUREMENT


def measure(index: int) -> Callable[[np.ndarray], int]:
    """Performs a partial measurement on a single qubit in a quantum register.
    
    Args:
        index (int): The index of the qubit to measure (0-based indexing)
    
    Returns:
        Callable[[np.ndarray], int]: A function that takes a quantum register
                                   and returns the measurement result (0 or 1)
    """
    def m(register: np.ndarray) -> int:
        n = register.ndim
        axis = tuple(range(index)) + tuple(range(index + 1, n))
        probs = np.sum(np.abs(register) ** 2, axis=axis)
        p = probs[0] / np.sum(probs)
        s = [slice(0, 2)] * n
        result = int(np.random.rand() > p)
        s[index] = slice(0, 1) if result == 1 else slice(1, 2)
        register[tuple(s)] = 0
        register *= 1 / np.linalg.norm(register)
        return result
    
    return m


def measure_all(collapse: bool = True) -> Callable[[np.ndarray], Tuple[int, ...]]:
    """Returns a function that performs a complete measurement of all qubits.
    
    Args:
        collapse (bool): If True, collapses the wavefunction to the measured state
    
    Returns:
        Callable[[np.ndarray], Tuple[int, ...]]: A function that takes a quantum
                                               register and returns a tuple of
                                               measured values (0s and 1s)
    """
    def m(register: np.ndarray) -> Tuple[int, ...]:
        r = register.ravel()
        r = r / np.linalg.norm(r)
        index = np.random.choice(range(len(r)), p=np.abs(r)**2)
        multiindex = np.unravel_index(index, register.shape)
        if collapse:
            register.fill(0)
            register[multiindex] = 1.0
        return multiindex
    
    return m


# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
def max_multiindex(register: np.ndarray) -> Tuple[int, ...]:
    """Finds the computational basis state with the largest amplitude.
    
    Args:
        register (np.ndarray): The quantum register to analyze
    
    Returns:
        Tuple[int, ...]: A tuple of integers (0s and 1s) representing
                        the basis state with the largest amplitude
    """
    return np.unravel_index(np.argmax(np.abs(register)), register.shape)


# TESTING

def test_unary_gates() -> None:
    """Run tests on all unary gates."""
    register = random_register(2)
    x = X(0) (register)
    y = Y(0) (register)
    z = Z(0) (register)
    s = S(0) (register)
    t = T(0) (register)
    x1 = X(1) (register)
    sn = SNOT(1) (register)
    sn2 = SNOT(1) (sn)
    hxh = H(0) (X(0) (H(0) (register)))
    hyh = H(0) (Y(0) (H(0) (register)))
    hzh = H(0) (Z(0) (H(0) (register)))
    hh = H(0) (H(0) (register))
    ss = S(0) (S(0) (register))
    tt = T(0) (T(0) (register))
    uu = R(0, np.pi/8) (R(0, np.pi/8) (register))
    assert np.allclose(hxh, z)
    assert np.allclose(hyh, -y)
    assert np.allclose(hh, register)
    assert np.allclose(ss, z)
    assert np.allclose(tt, s)
    assert np.allclose(hxh, z)
    assert np.allclose(hyh, -y)
    assert np.allclose(hzh, x)
    assert np.allclose(sn2, x1)
    assert np.allclose(uu, t)


def test_binary_gates() -> None:
    """Run tests on all binary gates."""
    inp = random_register(3)
    out1 = CNOT(0, 1) (inp)
    out2 = CNOT(1, 0) (out1)
    out3 = CNOT(0, 1) (out2)
    swap = SWAP(0, 1) (inp)
    assert np.allclose(out3, swap)
    not_gate = np.array([[0, 1], [1, 0]])
    out4 = C(0, 1, not_gate) (inp)
    assert np.allclose(out1, out4)


def test_ternary_gates() -> None:
    """Run tests on all ternary gates."""
    inp = random_register(4)
    assert np.allclose(inp, CCNOT(0,3,2) (CCNOT(0, 3, 2) (inp)))
    assert np.allclose(inp, CSWAP(0,3,2) (CSWAP(0, 3, 2) (inp)))


def test_bell_state() -> None:
    """Test the Bell state generator."""
    counts = Counter(
        measure_all() (bell_state(0,1) (quantum_register(3)))
        for _ in range(10000))
    assert counts[0, 0, 0] + counts[1, 1, 0] == 10000
    assert abs(counts[0, 0, 0] - counts[1, 1, 0]) < 500


def test_partial_measurement() -> None:
    """Run tests of the partial measurement function."""
    c = Counter()
    for _ in range(14000):
        register = np.array([[1, -2j], [3, 0]]) / 14 ** 0.5
        measure(0) (register)
        measure(1) (register)
        c[max_multiindex(register)] += 1
    assert abs(c[0, 0] - 1000) < 150
    assert abs(c[0, 1] - 4000) < 250
    assert abs(c[1, 0] - 9000) < 250
    assert c[1, 1] == 0
    m1 = m2 = 0
    for _ in range(14000):
        register = np.array([[1, -2j], [3, 0]]) / 14 ** 0.5
        m1 += measure(0) (register)
        m2 += measure(1) (register)
    assert abs(m1 - 9000) < 250
    assert abs(m2 - 4000) < 150


def test_full_measurement(num_trials = 100_000) -> None:
    register = np.array([0.3, 0.4j, -0.5, 0.5 ** 0.5]).reshape((2, 2))
    m = measure_all(collapse=False)
    expected = [num_trials * 0.09, num_trials * 0.16, num_trials * 0.25, num_trials * 0.5]
    counts = Counter(m(register) for _ in range(num_trials))
    observed = [counts[0,0], counts[0,1], counts[1,0], counts[1,1]]
    assert all(abs(x - y) < 400 for x, y in zip(expected, observed))


def test_circuits() -> None:
    register = random_register(2)
    bell1 = bell_state(0, 1)
    bell2 = circuit([H(0), CNOT(0, 1)])
    assert np.allclose(bell1(register), bell2(register))


# Test function to demonstrate usage
def test_display():
    """Test the display function with various quantum states."""
    # Create a simple 2-qubit state
    psi = np.array([[1, 2], [3, 4]], dtype=complex) / np.sqrt(30)
    print("Two-qubit state:")
    display(psi)

    # Create a superposition state
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    state = np.kron(hadamard, hadamard) / np.sqrt(2)
    print("\nTwo-qubit Hadamard state:")
    display(state)


def run_tests() -> None:
    """Run all tests."""
    test_unary_gates()
    test_binary_gates()
    test_ternary_gates()
    test_bell_state()
    test_partial_measurement()
    test_full_measurement()
    test_circuits()


def main():
    print('Running tests.')
    run_tests()
    print('Tests complete. No errors found.')


if __name__ == '__main__':
    main()
