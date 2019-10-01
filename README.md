# Quantum Simulator
## A quantum computer simulator using Python and NumPy.

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

# Credits

Written by David Radcliffe (dradcliffe@gmail.com)

# License

This code is licensed under the terms of the GNU Public License. See the LICENSE file for more details.
Please cite the author and link to the repository if you use or redistribute this code,
whether in original or modified form.
