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
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from typing import Sequence
from sklearn.svm import SVC
from qiskit_machine_learning.kernels import TrainableKernel
from qiskit.circuit.library import RXGate
def calc_fidelity_kernel_matrix(X1_dm,X2_dm):
    kernel_matrix=np.zeros((len(X1_dm),len(X2_dm)))
    for row in range(len(X1_dm)):
        for col in range(len(X2_dm)):
            kernel_matrix[row][col]=qi.state_fidelity(X1_dm[row],X2_dm[col],validate=True)
    return np.array(kernel_matrix)


class QSVC_Loss(SVCLoss):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments to pass to SVC constructor within
                      SVCLoss evaluation.
        """
        self.kwargs = kwargs

    def evaluate(
        self,
        parameter_values: Sequence[float],
        quantum_kernel,
        data,
        labels: np.ndarray,
    ) -> float:
        # Bind training parameters
        quantum_kernel.assign_training_parameters(parameter_values)
        results=[]
        for i in data:
            results.append(i.evolve(quantum_kernel.feature_map))

        # Get estimated kernel matrix
        #kmatrix = quantum_kernel.evaluate(data)
        kmatrix=calc_fidelity_kernel_matrix(results,results)

        # Train a quantum support vector classifier
        svc = SVC(kernel="precomputed", **self.kwargs)
        svc.fit(kmatrix, labels)

        # Get dual coefficients
        dual_coefs = svc.dual_coef_[0]

        # Get support vectors
        support_vecs = svc.support_

        # Prune kernel matrix of non-support-vector entries
        kmatrix = kmatrix[support_vecs, :][:, support_vecs]

        # Calculate loss
        loss = np.sum(np.abs(dual_coefs)) - (0.5 * (dual_coefs.T @ kmatrix @ dual_coefs))

        return loss
    
