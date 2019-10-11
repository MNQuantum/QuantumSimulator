#
# File:   teleport.qasm
# Date:   2019-10-10
# Author: David Radcliffe
#
# Implements quantum teleportation. The quantum state in q0 is transferred to q2.
#

  qubit q0,\psi
  qubit q1,0
  qubit q2,0

# Create an ancillary EPR pair
  h q1
  cnot q1,q2

# Perform a Bell measurement on the first two qubits.
  cnot q0,q1
  h q0
  nop q1
  measure q0
  measure q1

# Use the result of measurement to correct the third qubit.
  c-x q1,q2
  c-z q0,q2
