

from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import QuantumCircuit,QuantumRegister, Aer, execute
import numpy as np
import core

import preprocessing

def rx_kernel(qc,x_params,n_external_inputs,n_extra_qubits):
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)


def rx_circuit(x_params,n_inputs,n_extra_qubits,n_output):
    qc=QuantumCircuit(n_inputs,n_output)
    state_vector=Statevector(qc)
    qc.initialize(state_vector,list(range(0,n_inputs)))
    qc.barrier()
    rx_kernel(qc,x_params,n_inputs,n_extra_qubits)
    qc.barrier()
    qc.measure(list(range(n_output)),list(range(n_output)))
    return qc

def ising_rxx_rxz_interaction(qc,x_params,theta,n_inputs,n_layers,n_external_inputs,n_extra_qubits):
    typ=isinstance(theta,list)
    param_index=0
    for layer in range(n_layers):
        for i in range(n_external_inputs):
            qc.rx(x_params[i], i)
        for j in range(n_extra_qubits):
            qc.h(n_external_inputs+j)

        for q2 in range(1,n_inputs):
            qc.rzz(theta[param_index],0,q2)
            param_index+=1

        for q2 in range(1,n_inputs):
            qc.rzx(theta[param_index],0,q2)
            param_index+=1

        for q_tmp in range(n_inputs):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1
 
    for i in range(n_external_inputs):
        qc.rx(x_params[i], i)
    for j in range(n_extra_qubits):
        qc.h(n_external_inputs+j)

    return qc



def ising_interaction(qc,x_params,theta,n_layers,n_qubit):
    param_index=0
    for layer in range(n_layers):
        #for i in range(n_feature):
        #    qc.rx(x_params[i], i)

        for q1 in range(n_qubit):
            for q2 in range(q1,n_qubit):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(n_qubit):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1
 
    for i in range(n_qubit):
        qc.rx(x_params[i], i)
    for i in range(n_qubit):
        qc.rx(x_params[i+2], i)

    return qc




def ising_interaction_noInput(qc,theta,n_layers,n_qubit):
    typ=isinstance(theta,list)
    param_index=0
    for layer in range(n_layers):

        for q1 in range(n_qubit):
            for q2 in range(q1,n_qubit):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(n_qubit):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1

    return qc



