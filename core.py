import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.execute_function import execute
from qiskit.providers.aer import AerSimulator
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

    def __init__(self,encoder: QuantumCircuit,decoder: QuantumCircuit,num_encoder_params:int,trash_qubits,num_auxiliary_encoder,num_auxiliary_decoder,**kwargs):
            super(QVAE_NN, self).__init__(**kwargs)
            self._encoder = encoder.copy()
            #if len(self._encoder.clbits) == 0:
            #    self._encoder.measure_all()
            self._decoder = decoder.copy()
            self._num_encoder_params=num_encoder_params
            self._trash_qubits=trash_qubits
            self._num_auxiliary_encoder=num_auxiliary_encoder
            self._num_auxiliary_decoder=num_auxiliary_decoder

    def forward(self,input_data,weights):
        """
        Forward pass of the network.
        """
        num_encoder_params=self._num_encoder_params
        encoder_weights=weights[:num_encoder_params]
        decoder_weights=weights[num_encoder_params:]
        #encoder_weights=[0]*num_encoder_params
        #decoder_weights=[0]*(len(weights)-num_encoder_params)
        
        trash_qubits=self._trash_qubits
        _, num_samples = self._preprocess_forward(input_data, encoder_weights)

        ### Encoder 
        
        #encoder_params=[encoder_weights]*num_samples # overwrite parameter_values to remove in the input data
        n_qubit = math.log2(len(input_data[0]))
        assert(int(n_qubit)==n_qubit)
        n_qubit=int(n_qubit)
        original_encoder=self._encoder.copy('original_e')

        my_encoder=original_encoder.assign_parameters(encoder_weights)
        backend=AerSimulator(method='statevector')

        #qc_list=[]
        result_tmp=[]
        for i in range(num_samples):
            qc_e=QuantumCircuit(n_qubit+self._num_auxiliary_encoder)
            qc_e.initialize(input_data[i], range(n_qubit))
            # input + auxiliary? !!! check
            qc_e=qc_e.compose(my_encoder)
            state_vector = qi.Statevector.from_instruction(qc_e) 
            result_tmp.append(state_vector)

        # sampler allows batching
        #job = self.sampler.run(qc_list, encoder_params)

        #try:
        #    results_tmp = job.result()
        #except Exception as exc:
        #    raise QiskitMachineLearningError("Sampler job failed.") from exc
        #result_tmp = self._postprocess(num_samples, results_tmp) #full dimensional output # Sampler returns probabilities
        
        ### partial trace over auxiliary qubits encoder 
        result=[]
        for state in result_tmp:
            quantum_state=qi.Statevector(state)
            auxiliary_qubits_e=range(n_qubit,n_qubit+self._num_auxiliary_encoder)
            tmp_state=qi.partial_trace(quantum_state,auxiliary_qubits_e)
            trace=tmp_state.trace()
            tmp_state=tmp_state/trace
            result.append(tmp_state) # reduced output by tracing out auxiliary qubits
            
        if trash_qubits==None:
            reduced_result=result
        else:
            reduced_result=[]
            for quantum_state in result:
                reduced_state=qi.partial_trace(quantum_state,trash_qubits)
                trace=reduced_state.trace()
                reduced_state=reduced_state/trace
                reduced_result.append(reduced_state) # reduced output by tracing out trash qubits
                
        
        ### Decoder
        num_trash=2**len(trash_qubits)
        zero_state=np.zeros((num_trash,num_trash))
        zero_state[0][0]=1
        reconstruction_qubits=qi.DensityMatrix(zero_state)

        num_aux=2**self._num_auxiliary_decoder
        aux_state=np.zeros((num_aux,num_aux))
        aux_state[0][0]=1
        auxiliary_qubits=qi.DensityMatrix(aux_state)
        original_decoder=self._decoder.copy('original_d')
        qc_d=original_decoder.assign_parameters(decoder_weights,inplace=False)
        decoder_output=[]

        for item in reduced_result:
            latent_state=qi.DensityMatrix(item)
            latent_full=latent_state.tensor(reconstruction_qubits)
            #assert(2**n_qubit==latent_full.dim)
            latent_full=latent_full.tensor(auxiliary_qubits)
            ### partial trace over auxiliary qubits decoder 
            quantum_state=latent_full.evolve(qc_d)
            auxiliary_qubits_d=range(n_qubit,n_qubit+self._num_auxiliary_decoder)
            tmp_state=qi.partial_trace(quantum_state,auxiliary_qubits_d) # reduced output by tracing out auxiliary qubits
            trace=tmp_state.trace()
            tmp_state=tmp_state/trace
            decoder_output.append(tmp_state)
        #print('\nhj',qi.DensityMatrix(input_data[0]),decoder_output[0])
        return decoder_output,reduced_result


class QVAE_trainer(NeuralNetworkRegressor):
    def __init__(self,beta,divergence_type,reconstruction_loss='fidelity',**kwargs):
        super(QVAE_trainer, self).__init__(**kwargs)
        self._reconstruction_loss=reconstruction_loss
        self._beta=beta
        self._divergence_type=divergence_type

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        function: ObjectiveFunction = None
        #function = StateVector_ObjectiveFunction(X, y, self._neural_network, self._loss)
        function = DensityMatrix_ObjectiveFunction(X=X, y=y, neural_network=self._neural_network,loss=self._loss,reconstruction_loss=self._reconstruction_loss,beta=self._beta,divergence_type=self._divergence_type)
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
    def __init__(self,reconstruction_loss,beta,divergence_type,**kwargs):
        super(DensityMatrix_ObjectiveFunction, self).__init__(**kwargs)
        self._reconstruction_loss=reconstruction_loss
        self._beta=beta
        self._divergence_type=divergence_type

    def reconstruction_loss(self,matrix,vector):
        sum=0
        for v,m in zip(vector,matrix):
            if self._reconstruction_loss=='cross_entropy':
                m=qi.DensityMatrix(scipy.linalg.logm(m)/np.log(2.0))
                trace=m.trace()
                m=m/trace
                current_fidelity=qi.state_fidelity(m,v,validate=False) # m has difficulty getting trace one
            elif self._reconstruction_loss=='fidelity':
                current_fidelity=qi.state_fidelity(m,v,validate=True)
            else:
                raise ValueError('reconstruction loss type not recognized')

            sum=sum+current_fidelity
        return -sum
    
    def quantum_relative_entropy(self,latent):
        type_divergence=self._divergence_type
        entropy_loss=0
            
        ## analogy KL divergence
        if type_divergence=='KLD':
            for state in latent:
                dim=latent[0].dim
                max_mixed_state=np.diag(np.full(dim,1/dim))
                max_entropy=qi.entropy(max_mixed_state) ## this is log base 2 by default
                log_state=scipy.linalg.logm(state)/np.log(2.0) # in base 2
                relative_entropy=-qi.DensityMatrix(np.dot(max_mixed_state,log_state)).trace()-max_entropy #KL(a||b), a is actual, b is predict
                entropy_loss=entropy_loss-relative_entropy.real
            return entropy_loss

        ## analogy Jensen-Shannon divergence
        elif type_divergence=='JSD':
            for state in latent:
                dim=latent[0].dim
                max_mixed_state=np.diag(np.full(dim,1/dim))
                M_state=0.5*(state+max_mixed_state)
                log_M=scipy.linalg.logm(M_state)/np.log(2.0) # in base 2
                my_entropy=qi.entropy(state) 
                max_entryopy=qi.entropy(max_mixed_state)
                #JSD(a||b)=0.5*KL(a||M)+0.5*KL(b||M), M=0.5a+0.5b
                relative_entropy=(-qi.DensityMatrix(np.dot(state,log_M)).trace()-my_entropy-qi.DensityMatrix(np.dot(max_mixed_state,log_M)).trace()-max_entryopy)*0.5 ##check if correct!
                entropy_loss=entropy_loss+relative_entropy.real
            return entropy_loss
        else:
            raise ValueError('divergence type not recognized')

    def objective(self, weights: np.ndarray) -> float:
        # output is of shape (N, num_outputs)
        forward_result=self._neural_network_forward(weights)
        output = forward_result[0]
        latent=forward_result[1]
        
        val_1 =self.reconstruction_loss(matrix=output, vector=self._y)
        val_2 =self.quantum_relative_entropy(latent=latent)
        val = (val_1+self._beta*val_2) / self._num_samples
        return val
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def encoder(n_layers,n_qubit,theta_params,num_auxiliary_encoder):
    qc=QuantumCircuit(n_qubit+num_auxiliary_encoder)
    embedding.ising_interaction_noInput(qc,theta_params,n_layers,n_qubit,num_auxiliary_encoder) 
    return qc

def decoder(n_layers,n_qubit,theta_params,num_auxiliary_decoder):
    qc=QuantumCircuit(n_qubit+num_auxiliary_decoder)
    embedding.ising_interaction_noInput(qc,theta_params,n_layers,n_qubit,num_auxiliary_decoder) 
    return qc