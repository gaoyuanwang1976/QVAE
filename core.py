import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.execute_function import execute
import qiskit.quantum_info as qi
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import ObjectiveFunction
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
import scipy
import math
import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)
import embedding
import copy



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
        #_, num_samples = self._preprocess_forward(input_data, encoder_weights)
        num_samples=len(input_data)

        ### Encoder
        n_qubit = math.log2(len(input_data[0]))
        assert(int(n_qubit)==n_qubit)
        n_qubit=int(n_qubit)
        original_encoder=self._encoder.copy('original_e')
        my_encoder=original_encoder.assign_parameters(encoder_weights)

        '''
        ### state vector version encoder 
        #qc_list=[]
        result_tmp=[]
        for i in range(num_samples):
            qc_e=QuantumCircuit(n_qubit+self._num_auxiliary_encoder)
            qc_e.initialize(input_data[i], range(n_qubit))
            # input + auxiliary? !!! check
            qc_e=qc_e.compose(my_encoder)
            state_vector = qi.Statevector.from_instruction(qc_e) 
            result_tmp.append(state_vector)
        
        ### partial trace over auxiliary qubits encoder 
        result=[]
        for state in result_tmp:
            quantum_state=qi.Statevector(state)
            auxiliary_qubits_e=range(n_qubit,n_qubit+self._num_auxiliary_encoder)
            tmp_state=qi.partial_trace(quantum_state,auxiliary_qubits_e)
            trace=tmp_state.trace()
            tmp_state=tmp_state/trace
            result.append(tmp_state) # reduced output by tracing out auxiliary qubits
        '''

        ### DensityMatrix version encoder 
        result_tmp=[]
        for i in range(num_samples):
            num_aux_en=2**self._num_auxiliary_encoder
            aux_state_en=np.zeros((num_aux_en,num_aux_en))
            aux_state_en[0][0]=1
            auxiliary_qubits_en=qi.DensityMatrix(aux_state_en)
            my_state=qi.DensityMatrix(input_data[i])
            my_state=my_state.tensor(auxiliary_qubits_en)
            my_state=my_state.evolve(my_encoder)
            result_tmp.append(my_state)


        result=[]
        for state in result_tmp:
            auxiliary_qubits_e=range(n_qubit,n_qubit+self._num_auxiliary_encoder)
            tmp_state=qi.partial_trace(state,auxiliary_qubits_e)
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
        return decoder_output,reduced_result

class QVAE_trainer(NeuralNetworkRegressor):
    def __init__(self,beta,regularizer_type,reconstruction_loss='fidelity',**kwargs):
        super(QVAE_trainer, self).__init__(**kwargs)
        self._reconstruction_loss=reconstruction_loss
        self._beta=beta
        self._regularizer_type=regularizer_type
        #self._global_state_flag=global_state_flag

    def _fit_internal(self, X: np.ndarray, y: np.ndarray):
        function: ObjectiveFunction = None
        #function = StateVector_ObjectiveFunction(X, y, self._neural_network, self._loss)

        function = DensityMatrix_ObjectiveFunction(X=X, y=y, neural_network=self._neural_network,loss=self._loss,reconstruction_loss=self._reconstruction_loss,beta=self._beta,regularizer_type=self._regularizer_type)
        return self._minimize(function)
    
    def score_fidelity(self, X, y):
        y_pred = self.predict(X)[0]
        fidelity_score=0
        for i,j in zip(y_pred,y):
            fidelity_score =fidelity_score+qi.state_fidelity(i,j,validate=True)
        return fidelity_score/len(y)
    
    def score(self, X, y, loss_type):
        y_pred = self.predict(X)[0]
        if loss_type=='fidelity':
            score=0
            for i,j in zip(y_pred,y):
                score =score+qi.state_fidelity(i,j,validate=True)
            return score/len(y)
        
        elif loss_type=='wasserstein': 
            num_data=len(X)

            dim=len(X[0])
            n_qubits_latent=int(2*math.log2(dim))
            swap_circuit = QuantumCircuit(n_qubits_latent,0)
            for i in range(int(n_qubits_latent*0.5)):
                swap_circuit.swap(i, int(n_qubits_latent*0.5)+i)
            backend = Aer.get_backend('unitary_simulator')
            job=execute(swap_circuit,backend,shots=100)
            result=job.result()
            swap=result.get_unitary(swap_circuit,3).data
            identity_matrix=np.identity(dim*dim)
            C=0.5*(identity_matrix-swap)

            cost=0
            pi_state=np.zeros((len(X[0])*len(X[0]),len(X[0])*len(X[0])))
            for v,m in zip(y,y_pred):
                input_densityMatrix=qi.DensityMatrix(v)
                in_out_state=qi.DensityMatrix(m).expand(input_densityMatrix)
                pi_state=(pi_state+in_out_state.data*1./num_data)
            cost=np.trace(np.matmul(pi_state,C))
            return -cost.real
        
        elif loss_type=='cross_entropy':
            score=0
            for v,m in zip(y,y_pred):
                input_state_entropy=qi.entropy(v)
                log_state=qi.DensityMatrix(scipy.linalg.logm(m/np.log(2.0)))
                relative_entropy=-qi.DensityMatrix(np.dot(v,log_state)).trace()-input_state_entropy
                score+=relative_entropy.real
            return -score/len(y)
        
        elif loss_type=='JSD':
            score=0
            for v,m in zip(y,y_pred):
                M_state=0.5*(v.data+m.data)
                log_M=qi.DensityMatrix(scipy.linalg.logm(M_state)/np.log(2.0)) # in base 2
                relative_entropy=(-qi.DensityMatrix(np.dot(v,log_M)).trace()-qi.entropy(v)-qi.DensityMatrix(np.dot(m,log_M)).trace()-qi.entropy(m))*0.5 ##check if correct!
                score+=relative_entropy.real
            return -score*1./len(y)
        else:
            raise ValueError('reconstruction loss type not recognized')

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
    def __init__(self,reconstruction_loss,beta,regularizer_type,**kwargs):
        super(DensityMatrix_ObjectiveFunction, self).__init__(**kwargs)
        self._reconstruction_loss=reconstruction_loss
        self._beta=beta
        self._regularizer_type=regularizer_type
        #self._global_state_flag=global_state_flag

    def reconstruction_loss(self,matrix,vector):
        if self._reconstruction_loss=='wasserstein':
            num_data=len(vector)

            dim=len(vector[0])
            n_qubits_latent=int(2*math.log2(dim))
            swap_circuit = QuantumCircuit(n_qubits_latent,0)
            for i in range(int(n_qubits_latent*0.5)):
                swap_circuit.swap(i, int(n_qubits_latent*0.5)+i)
            backend = Aer.get_backend('unitary_simulator')
            job=execute(swap_circuit,backend,shots=100)
            result=job.result()
            swap=result.get_unitary(swap_circuit,3).data
            identity_matrix=np.identity(dim*dim)
            C=0.5*(identity_matrix-swap)

            cost=0
            pi_state=np.zeros((len(vector[0])*len(vector[0]),len(vector[0])*len(vector[0])))
            for v,m in zip(vector,matrix):
                input_densityMatrix=qi.DensityMatrix(v)
                in_out_state=qi.DensityMatrix(m).expand(input_densityMatrix)
                pi_state=(pi_state+in_out_state.data*1./num_data)
            cost=np.trace(np.matmul(pi_state,C))
            return cost.real
        
        else:
            if self._reconstruction_loss=='cross_entropy':
                sum=0
                for v,m in zip(vector,matrix):
                    input_state_entropy=qi.entropy(v)
                    log_state=qi.DensityMatrix(scipy.linalg.logm(m)/np.log(2.0))
                    relative_entropy=-qi.DensityMatrix(np.dot(v,log_state)).trace()-input_state_entropy
                    #current_fidelity=qi.state_fidelity(m,v,validate=False) # this is equal to the first term in relative entropy only if v is a pure state
                    sum+=relative_entropy.real
                return sum*1./len(vector)

            elif self._reconstruction_loss=='fidelity':
                sum=0
                for v,m in zip(vector,matrix):
                    current_fidelity=qi.state_fidelity(m,v,validate=True)
                    sum+=current_fidelity
                return -sum*1./len(vector)
            
            elif self._reconstruction_loss=='JSD':
                sum=0
                for v,m in zip(vector,matrix):
                    M_state=0.5*(v.data+m.data)
                    log_M=scipy.linalg.logm(M_state)/np.log(2.0) # in base 2
                    relative_entropy=(-qi.DensityMatrix(np.dot(v,log_M)).trace()-qi.entropy(v)-qi.DensityMatrix(np.dot(m,log_M)).trace()-qi.entropy(m))*0.5 ##check if correct!
                    sum+=relative_entropy.real
                return sum*1./len(vector)

            else:
                raise ValueError('reconstruction loss type not recognized')

        
    def quantum_relative_entropy(self,latent):
        type_regularizer=self._regularizer_type
        dim=latent[0].dim
        max_mixed_state=np.diag(np.full(dim,1/dim))
        ## analogy KL divergence
        if type_regularizer=='KLD':
            '''
            if self._global_state_flag==True:
                combined_state=np.zeros((len(latent[0].data),len(latent[0].data)))
                for state in latent:
                    combined_state=combined_state+state.data*1./len(latent)
                max_entropy=qi.entropy(max_mixed_state) ## this is log base 2 by default
                log_state=scipy.linalg.logm(combined_state)/np.log(2.0) # in base 2
                relative_entropy=-qi.DensityMatrix(np.dot(max_mixed_state,log_state)).trace()-max_entropy #KL(a||b), a is actual, b is predict. relative_entropy>=0
                entropy_loss=relative_entropy.real
                return entropy_loss
            else:
            '''
            entropy_loss=0
            for state in latent:
                max_entropy=qi.entropy(max_mixed_state) ## this is log base 2 by default
                log_state=scipy.linalg.logm(state)/np.log(2.0) # in base 2
                relative_entropy=-qi.DensityMatrix(np.dot(max_mixed_state,log_state)).trace()-max_entropy #KL(a||b), a is actual, b is predict
                entropy_loss=entropy_loss+relative_entropy.real
            return entropy_loss*1./len(latent)

        ## analogy Jensen-Shannon divergence
        elif type_regularizer=='JSD':
            '''
            if self._global_regularizer_flag==True:
                combined_state=np.zeros((len(latent[0].data),len(latent[0].data)))
                for state in latent:
                    combined_state=combined_state+state.data*1./len(latent)

                M_state=0.5*(state+max_mixed_state)
                log_M=scipy.linalg.logm(M_state)/np.log(2.0) # in base 2
                my_entropy=qi.entropy(state) 
                max_entryopy=qi.entropy(max_mixed_state)
                #JSD(a||b)=0.5*KL(a||M)+0.5*KL(b||M), M=0.5a+0.5b
                relative_entropy=(-qi.DensityMatrix(np.dot(state,log_M)).trace()-my_entropy-qi.DensityMatrix(np.dot(max_mixed_state,log_M)).trace()-max_entryopy)*0.5 ##check if correct!
                entropy_loss=relative_entropy.real
                return entropy_loss

            else: 
            '''
            entropy_loss=0       
            for state in latent:
                M_state=0.5*(state+max_mixed_state)
                log_M=scipy.linalg.logm(M_state)/np.log(2.0) # in base 2
                my_entropy=qi.entropy(state) 
                max_entryopy=qi.entropy(max_mixed_state)
                #JSD(a||b)=0.5*KL(a||M)+0.5*KL(b||M), M=0.5a+0.5b
                relative_entropy=(-qi.DensityMatrix(np.dot(state,log_M)).trace()-my_entropy-qi.DensityMatrix(np.dot(max_mixed_state,log_M)).trace()-max_entryopy)*0.5 ##check if correct!
                entropy_loss=entropy_loss+relative_entropy.real
            return entropy_loss*1./len(latent)

        elif type_regularizer=='fidelity':
            '''
            if self._global_regularizer_flag==True:
                combined_state=np.zeros((len(latent[0].data),len(latent[0].data)))
                for state in latent:
                    combined_state=combined_state+state.data*1./len(latent)
                entropy_loss=qi.state_fidelity(combined_state,max_mixed_state,validate=True)
                return -entropy_loss

            else:
            '''
            entropy_loss=0
            for state in latent:
                entropy_loss=entropy_loss+qi.state_fidelity(state,max_mixed_state,validate=True)
            return -entropy_loss*1./len(latent)
        
        elif type_regularizer=='wasserstein': #this definition of \pi is bad for two mixed states.
            dim=latent[0].dim
            max_mixed_state=np.diag(np.full(dim,1/dim))

            n_qubits_latent=int(2*math.log2(dim))
            swap_circuit = QuantumCircuit(n_qubits_latent,0)
            for i in range(int(n_qubits_latent*0.5)):
                swap_circuit.swap(i, int(n_qubits_latent*0.5)+i)
            backend = Aer.get_backend('unitary_simulator')
            job=execute(swap_circuit,backend,shots=100)
            result=job.result()
            swap=result.get_unitary(swap_circuit,3).data
            identity_matrix=np.identity(dim*dim)
            C=0.5*(identity_matrix-swap)
            '''
            if self._global_regularizer_flag==True:
                combined_state=np.zeros((len(latent[0].data),len(latent[0].data)))
                for state in latent:
                    combined_state=combined_state+state.data*1./len(latent)
                pi_state=(qi.DensityMatrix(max_mixed_state).expand(combined_state)).data
                cost=np.trace(np.matmul(pi_state,C))
                return cost.real

            else:
            '''
            cost=0
            for state in latent:          
                pi_state=(qi.DensityMatrix(max_mixed_state).expand(state)).data
                cost=cost+np.trace(np.matmul(pi_state,C))/len(latent)
            return cost.real
        else:
            raise ValueError('divergence type not recognized')

    def objective(self, weights: np.ndarray) -> float:
        # output is of shape (N, num_outputs)
        forward_result=self._neural_network_forward(weights)
        output = forward_result[0]
        latent=forward_result[1]
        
        val_1 =self.reconstruction_loss(matrix=output, vector=self._y)
        val_2 =self.quantum_relative_entropy(latent=latent)
        val = val_1-self._beta*val_2 #minimize, normalization is in val_1 and val_2
        return val
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def encoder(n_layers,n_qubit,theta_params,num_auxiliary_encoder,input_task):
    qc=QuantumCircuit(n_qubit+num_auxiliary_encoder)
    if input_task=='gene':
        embedding.ising_interaction_noInput(qc,theta_params,n_layers,n_qubit,num_auxiliary_encoder)
    elif input_task=='mnist':
        embedding.mnist_circuit(qc,theta_params,n_layers,n_qubit)
    return qc

def decoder(n_layers,n_qubit,theta_params,num_auxiliary_decoder,input_task):
    qc=QuantumCircuit(n_qubit+num_auxiliary_decoder)
    if input_task=='gene':
        embedding.ising_interaction_noInput(qc,theta_params,n_layers,n_qubit,num_auxiliary_decoder)
    elif input_task=='mnist':
        embedding.mnist_circuit(qc,theta_params,n_layers,n_qubit).inverse()
    return qc