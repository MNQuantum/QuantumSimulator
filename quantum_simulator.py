"""
Quantum computing simulator using NumPy tensors.

An n-qubit quantum register is represented by a tensor of complex numbers
of shape (2, 2, ..., 2), containing 2^n complex numbers. Each complex number
requires 64 bits of memory. The sum of the squared magnitudes of a quantum
register is always one.

Quantum gates are also represented as complex tensors.
  *  A unary gate is a tensor of shape (2, 2).
  *  A binary gate is a tensor of shape (2, 2, 2, 2).
  *  A ternary gate is a tensor of shape (2, 2, 2, 2, 2, 2).
Quantum gates are applied using tensor contraction, which is a
generalization of matrix multiplication.

Any quantum circuit can be expressed using only unary and binary gates.
Ternary gates are rarely used in practice since they are expensive to compute.

In this package, the quantum gate functions are "higher-level functions". 
The return value is always another function, which can be applied to a quantum register.
This makes it easier to reuse gates in a circuit.

Measuring an n-qubit quantum register returns a binary vector of length n,
representing a random multi-index of the tensor. The probability of obtaining
a multi-index is equal to the squared magnitude of the corresponding entry
of the tensor. Measuring a register changes the state of the register.

Author: David Radcliffe (dradcliffe@gmail.com)

This file is open source under the GPL License. See the LICENSE file for more details.
If you use the code in any way, please credit the author and include a link to the
repository: https://github.com/MNQuantum/QuantumSimulator.

Last updated: 2019-09-30
"""

from itertools import product
import numpy as np
from collections import Counter


def quantum_register(number_of_qubits):
    """Creates a linear register of qubits, initialized to |000...0>.
    Returns a complex tensor of shape (2, 2, ..., 2)."""
    shape = (2,) * number_of_qubits
    first = (0,) * number_of_qubits
    register = np.zeros(shape, dtype=np.complex64)
    register[first] = 1+0j
    return register


def random_register(number_of_qubits):
    """Creates a linear register of qubits with random complex amplitudes.
    Returns a complex tensor of shape (2, 2, ..., 2)."""
    shape = (2,) * number_of_qubits
    register = np.random.rand(*shape) + 1j * np.random.rand(*shape) - (0.5 + 0.5j)
    register = register / np.linalg.norm(register)
    return register


def display(tensor):
    """Displays the tensor entries in tabular form"""
    for multiindex in product(range(2), repeat=tensor.ndim):
        ket = '|' + str(multiindex)[1:-1].replace(', ', '') + '⟩'
        value = tensor[multiindex]
        print('%s\t%.5f + %.5f i' % (ket, value.real, value.imag))


def get_transposition(n, indices):
    """Helper function that reorders a tensor after a quantum gate is applied."""
    transpose = [0] * n
    k = len(indices)
    ptr = 0
    for i in range(n):
        if i in indices:
            transpose[i] = n - k + indices.index(i)
        else:
            transpose[i] =  ptr
            ptr += 1
    return transpose


def apply_gate(gate, *indices):
    """Applies a gate to one or more indices of a quantum register.
    This is a higher-order function: it returns another function that
    can be applied to a register."""
    axes = (indices, range(len(indices)))
    def op(register):
        return np.tensordot(register, gate, axes=axes).transpose(
               get_transposition(register.ndim, indices))
    return op


def circuit(ops):
    """Constructs a circuit as a sequence of quantum gates.
    This higher-order function returns another function that
    can be applied to a quantum register."""
    def circ(register):
        for op in ops:
            register = op(register)
        return register
    return circ


def measure_circuit(ops, index=None):
    """Constructs a circuit and performs a measurement."""
    circ = circuit(ops)
    def m(register):
        if index is None:
            return measure_all() (circ(register))
        return measure(index) (circ(register))
    return m

##### Unary gates

def X(index):
    """Generates a Pauli X gate (also called a NOT gate) acting on a given index.
    It returns a function that can be applied to a register."""
    gate = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    return apply_gate(gate, index)


def Y(index):
    """Generates a Pauli Y gate acting on a given index.
    It returns a function that can be applied to a register."""
    gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    return apply_gate(gate, index)


def Z(index):
    """Generates a Pauli Z gate acting on a given index.
    This is the same as a rotation of the Bloch sphere by pi radians about the Z-axis.
    It returns a function that can be applied to a register."""
    gate = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    return apply_gate(gate, index)


def R(index, angle):
    """Generates a rotation of the Block sphere about the Z-axis by a given 
    It returns a function that can be applied to a register."""
    gate = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=np.complex64)
    return apply_gate(gate, index)


def S(index):
    """The S gate is a 90 degree rotation of the Bloch sphere about the Z-axis."""
    return R(index, np.pi/2)


def T(index):
    """The T gate is a 45 degree rotation of the Bloch sphere about the Z-axis.""" 
    return R(index, np.pi/4)


def H(index):
    """Generates a Hadamard gate. It returns a function that can be applied to a register."""
    r = np.sqrt(0.5)
    gate = np.array([[r, r], [r, -r]])
    return apply_gate(gate, index)


def SNOT(index):
    """Generates a 'square root of NOT' gate. It returns a function that can be applied to a register."""
    gate = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
    return apply_gate(gate, index)


### Binary gates


def SWAP(i, j):
    """Generates a SWAP gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 0, 1, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def CNOT(i, j):
    """Generates a controlled NOT gate, also called a controlled X gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 1, 0]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate,i, j)

CX = CNOT


def CY(i, j):
    """Generates a controlled Y gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, -1j,
                     0, 0, 1j, 0]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def CZ(i, j):
    """Generates a controlled Z gate."""
    gate = np.array([1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, -1]
                     ).reshape((2, 2, 2, 2))
    return apply_gate(gate, i, j)


def C(i, j, unary_gate):
    """Generates a controlled (binary) version of a unary gate.
    When the value at the first index is 1, apply the unary gate to the value at the second index.
    When the value at the first index is 0, do nothing."""
    gate = np.zeros((2, 2, 2, 2))
    gate[0, :, 0, :] = np.eye(2)
    gate[1, :, 1, :] = unary_gate
    return apply_gate(gate, i, j)


## Ternary gates

def CCNOT(i, j, k):
    """Generates a Toffoli gate."""
    gate = np.eye(8)
    gate[6:8, 6:8] = np.array([[0, 1], [1, 0]])
    gate = gate.reshape((2, 2, 2, 2, 2, 2))
    return apply_gate(gate, i, j, k)


def CSWAP(i, j, k):
    """Generates a Fredkin gate, or a controlled SWAP gate."""
    gate = np.eye(8)
    gate[5:7, 5:7] = np.array([[0, 1], [1, 0]])
    gate = gate.reshape((2, 2, 2, 2, 2, 2))
    return apply_gate(gate, i, j, k)

## Quantum circuits

def bell_state(i, j):
    """Generates an entangled Bell state on two qubits."""
    def b(register):
        return CNOT(i, j) (H(i) (register))
    return b


## MEASUREMENT

def measure(index):
    """Performs a partial measurement on a particular index.
    Returns a function that, when applied to a register,
    partially collapses the quantum state, and returns 0 or 1."""
    def m(register):
        n = register.ndim
        axis = tuple(range(index)) + tuple(range(index + 1, n))
        probs = np.sum(np.abs(register) ** 2, axis=axis)
        p = probs[0] / np.sum(probs)
        s = [slice(0, 2)] * n
        result = np.random.rand() > p
        s[index] = slice(0,1) if result else slice(1, 2)
        register[tuple(s)] = 0
        register *= 1 / np.linalg.norm(register)
        return result
    return m


def measure_all():
    """Returns a function that performs a full measurement on a quantum register, 
    fully collapsing the quantum state, and returns a binary vector of length equal
    to the number of qubits in the register."""
    def m(register):
        target = np.random.rand() * np.linalg.norm(register) ** 2 
        cumsum = 0.
        for multiindex in product(range(2), repeat=register.ndim):
            cumsum += np.abs(register[multiindex]) ** 2
            if cumsum >= target:
                break
        register.fill(0)
        register[(0,) * register.ndim] = 1
        return multiindex
    return m


# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
def max_multiindex(register):
    """Locates the entry of a tensor having the largest magnitude.
    Returns a binary vector of length equal to the number of qubits in the register.""" 
    return np.unravel_index(np.argmax(np.abs(register)), register.shape)


# TESTING

def test_unary_gates():
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


def test_binary_gates():
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


def test_ternary_gates():
    """Run tests on all ternary gates."""
    inp = random_register(4)
    assert np.allclose(inp, CCNOT(0,3,2) (CCNOT(0, 3, 2) (inp)))
    assert np.allclose(inp, CSWAP(0,3,2) (CSWAP(0, 3, 2) (inp)))


def test_bell_state():
    """Test the Bell state generator."""
    counts = Counter(
        measure_all() (bell_state(0,1) (quantum_register(3))) 
        for _ in range(10000))
    assert counts[0, 0, 0] + counts[1, 1, 0] == 10000
    assert abs(counts[0, 0, 0] - counts[1, 1, 0]) < 500


def test_measurement():
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

def test_circuits():
    register = random_register(2)
    bell1 = bell_state(0, 1)
    bell2 = circuit([H(0), CNOT(0, 1)])
    assert np.allclose(bell1(register), bell2(register))


def run_tests():
    """Run all tests."""
    test_unary_gates()
    test_binary_gates()
    test_ternary_gates()
    test_bell_state()
    test_measurement()
    test_circuits()


if __name__ == '__main__':
    print('Running tests.')
    run_tests()
    print('Tests complete. No errors found.')
