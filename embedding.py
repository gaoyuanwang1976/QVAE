

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

def ising_interaction_noInput(qc,theta,n_layers,n_qubit,num_auxiliary):
    typ=isinstance(theta,list)
    param_index=0
    num_qubit_total=n_qubit+num_auxiliary
    for layer in range(n_layers):

        for q1 in range(num_qubit_total):
            for q2 in range(q1,num_qubit_total):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(num_qubit_total):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1

    return qc

def mnist_circuit(circuit,parameters,n_layers,n_qubit):
    num_per_layer=10
    for layer in range(n_layers):
        #Applying a ry gate in each qubit
        for i in range(n_qubit):
            #the rotation of the ry gate is defined in the parameters list
            #based on the layer
            circuit.ry(parameters[(layer*num_per_layer)+i], i)
        circuit.barrier() #Create a barrier

        circuit.cx(2,0) #Apply a CNOT gate between the qubit 2 and 0
        circuit.cx(3,1) #Apply a CNOT gate between the qubit 3 and 1
        circuit.cx(5,4) #Apply a CNOT gate between the qubit 5 and 4
        circuit.barrier() #Create a barrier
        
        #Apply a RY gate in the qubit 0 with the rotation specified in the parameter list
        circuit.ry(parameters[6+(layer*num_per_layer)],0)
        #Apply a RY gate in the qubit 1 with the rotation specified in the parameter list
        circuit.ry(parameters[7+(layer*num_per_layer)],1)
        #Apply a RY gate in the qubit 4 with the rotation specified in the parameter list
        circuit.ry(parameters[8+(layer*num_per_layer)],4)
        circuit.barrier() #Create a barrier
        
        circuit.cx(4,1) #Apply a CNOT gate between the qubit 4 and 1
        circuit.barrier() #Create a barrier
        
        #Apply a RY gate in the qubit 1 with the rotation specified in the parameter list
        circuit.ry(parameters[9+(layer*num_per_layer)], 1)
        circuit.barrier() #Create a barrier
    return circuit
