#
# File:   swap.qasm
# Date:   2019-08-08
# Author: David Radcliffe
#
# Implements a SWAP gate using 3 CNOT gates
#

  qubit 	q0
  qubit 	q1

  cnot q0,q1
  cnot q1,q0
  cnot q0,q1
