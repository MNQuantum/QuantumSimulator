"""
Quantum teleportation
"""

import numpy as np
import quantum_simulator as q


def teleport(i, j, k):
    """
    Transfers a quantum state from i to k, assuming that j and k have been initialized to |0>.
    """
    def t(register):
        # Create an EPR pair using qubits j and k
        register = q.CNOT(j, k) (q.H(j) (register))

        # Perform a Bell measurement on qubits i and j
        register = q.H(i) (q.CNOT(i, j) (register))
        bit_i = q.measure(i) (register)
        bit_j = q.measure(j) (register)

        # Apply X and Z gates based on the values of the classical bits.
        if bit_i: 
            register = q.X(k) (register)
        if bit_j:
            register = q.Z(k) (register)
        
        return bit_i, bit_j, register
    return t


def teleport_test():
    for _ in range(100):
        # Create a 3-qubit register. The first qubit has a random state,
        # and the other two are initialized to |00>.
        register = q.quantum_register(3)
        register[:, 0, 0] = q.random_register(1)

        # Attempt teleportation
        b0, b1, observed = teleport(0, 1, 2)(register)

        # Confirm that the state of the first qubit has been transferred to the third qubit.
        expected = q.quantum_register(3)
        expected[0, 0, 0] = 0
        expected[b0, b1, :] = register[:, 0, 0]
        np.allclose(expected, observed)


if __name__ == '__main__':
    teleport_test()

