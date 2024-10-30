# Quantum Simulator
## A quantum computer simulator using Python and NumPy.

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

# Requirements

This script was developed using Python 3.6.8 and NumPy 1.22.0.
It should work with any version of Python from 3.5 or later.

# Credits

Written by David Radcliffe (dradcliffe@gmail.com).

# License

This code is licensed under the terms of the GNU Public License. See the LICENSE file for more details.
Please cite the author and link to the repository if you use or redistribute this code,
whether in original or modified form.
