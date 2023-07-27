import numpy as np
from qiskit import Aer, QuantumCircuit,execute
from qiskit.providers.aer import QasmSimulator
import qiskit.quantum_info as qi
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.algorithms import ObjectiveFunction
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
import scipy

import math
import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)
import embedding



class QVAE_NN(SamplerQNN):

    def __init__(self,encoder: QuantumCircuit,decoder: QuantumCircuit,num_encoder_params:int,trash_qubits,**kwargs):
            super(QVAE_NN, self).__init__(**kwargs)
            self._encoder = encoder.copy()
            if len(self._encoder.clbits) == 0:
                self._encoder.measure_all()
            self._decoder = decoder.copy()
            #if len(self._decoder.clbits) == 0:
            #    self._decoder.measure_all()
            self._num_encoder_params=num_encoder_params
            self._trash_qubits=trash_qubits

    def forward(self,input_data,weights):
        """
        Forward pass of the network.
        """
        num_encoder_params=self._num_encoder_params
        encoder_weights=weights[:num_encoder_params]
        decoder_weights=weights[num_encoder_params:]
        trash_qubits=self._trash_qubits
        _, num_samples = self._preprocess_forward(input_data, encoder_weights)

        ### Encoder 
        encoder_params=[encoder_weights]*num_samples # overwrite parameter_values to remove in the input data
        n_qubit = math.log2(len(input_data[0]))
        assert(int(n_qubit)==n_qubit)
        n_qubit=int(n_qubit)
        original_encoder=self._encoder.copy('original_e')

        qc_list=[]
        for i in range(num_samples):
            qc_e=QuantumCircuit(n_qubit)
            qc_e.initialize(input_data[i], range(n_qubit))
            qc_e=qc_e.compose(original_encoder)
            qc_list.append(qc_e)

        # sampler allows batching
        ### replace [self._circuit] * num_samples with qc_list
        job = self.sampler.run(qc_list, encoder_params)

        try:
            results = job.result()
        except Exception as exc:
            raise QiskitMachineLearningError("Sampler job failed.") from exc
        result = self._postprocess(num_samples, results) #full dimensional output 
        if trash_qubits==None:
            reduced_result=result
            #return result
        else:
            reduced_result=[]
            for state in result:
                quantum_state=qi.Statevector(state)
                reduced_state=qi.partial_trace(quantum_state,trash_qubits)
                trace=reduced_state.trace()
                reduced_state=reduced_state/trace
                reduced_result.append(reduced_state) # reduced output by tracing out trash qubits
            #return reduced_result

        ### Decoder
        num_trash=2**len(trash_qubits)
        zero_state=np.zeros((num_trash,num_trash))
        zero_state[0][0]=1
        reconstruction_qubits=qi.DensityMatrix(zero_state)
        original_decoder=self._decoder.copy('original_d')
        qc_d=original_decoder.assign_parameters(decoder_weights,inplace=False)
        decoder_output=[]

        for item in reduced_result:
            latent_state=qi.DensityMatrix(item)
            latent_full=latent_state.tensor(reconstruction_qubits)

            assert(2**n_qubit==latent_full.dim)
            decoder_output.append(latent_full.evolve(qc_d))

        return decoder_output,reduced_result


class QVAE_trainer(NeuralNetworkRegressor):
    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        function: ObjectiveFunction = None
        #function = StateVector_ObjectiveFunction(X, y, self._neural_network, self._loss)
        function = DensityMatrix_ObjectiveFunction(X, y, self._neural_network, self._loss)
        return self._minimize(function)
    
    def score(self, X, y):
        y_pred = self.predict(X)[0]
        fidelity_score=0
        for i,j in zip(y_pred,y):
            fidelity_score =fidelity_score+qi.state_fidelity(i,j,validate=True)

        return fidelity_score/len(y)

class StateVector_ObjectiveFunction(ObjectiveFunction):

    def objective(self, weights: np.ndarray) -> float:
        # output is of shape (N, num_outputs)
        output = self._neural_network_forward(weights)[0]
        val =sum(self._loss(output, self._y))
        val = val / self._num_samples
        return val
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:

        # weight probability gradient is of shape (N, num_outputs, num_weights)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)
        output = self._neural_network_forward(weights)[0]
        grad = np.zeros((1, self._neural_network.num_weights))
        num_outputs = self._neural_network.output_shape[0]

        for i in range(num_outputs):
            grad += weight_prob_grad[:, i, :].T @ self._loss(output[:,i], self._y[:,i])
            #grad += weight_prob_grad[:, i, :].T @ self._loss(np.full(num_samples, i), self._y)
        grad = grad / self._num_samples
        return grad
    
class DensityMatrix_ObjectiveFunction(ObjectiveFunction):

    def fidelity_loss(self,matrix,vector):
        sum=0
        for v,m in zip(vector,matrix):
            current_fidelity=qi.state_fidelity(m,v,validate=True) # check why is m not a valid state
            sum=sum+current_fidelity
        return -sum
    
    def quantum_relative_entropy(self,latent):
        dim=latent[0].dim
        max_mixed_state=np.diag(np.full(dim,1/dim))
        log_m=scipy.linalg.logm(max_mixed_state)/np.log(2.0) # must be in base 2
        entropy_loss=0
        for state in latent:
            my_entropy=qi.entropy(state) ## this is log base 2 by default
            relative_entropy=-qi.DensityMatrix(np.dot(state,log_m)).trace()-my_entropy
            entropy_loss=entropy_loss+relative_entropy.real
        return entropy_loss

    def objective(self, weights: np.ndarray) -> float:
        # output is of shape (N, num_outputs)
        forward_result=self._neural_network_forward(weights)
        output = forward_result[0]
        latent=forward_result[1]
        #val =sum(self._loss(output, self._y))
        val_1 =self.fidelity_loss(matrix=output, vector=self._y)
        val_2 =self.quantum_relative_entropy(latent=latent)

        val = (val_1+val_2) / self._num_samples
        return val
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def encoder(n_layers,n_qubit,theta_params):
    qc=QuantumCircuit(n_qubit)
    embedding.ising_interaction_noInput(qc,theta_params,n_layers,n_qubit) 
    return qc

def decoder(n_layers,n_qubit,theta_params):
    qc=QuantumCircuit(n_qubit)
    embedding.ising_interaction_noInput(qc,theta_params,n_layers,n_qubit) 
    return qc