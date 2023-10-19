import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM, TNC
from qiskit.providers.aer import AerSimulator

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

from sklearn.metrics import accuracy_score
import preprocessing
import QSVC_core

import math
import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)

import argparse

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering


# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=10)
    parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=5)
    parser.add_argument('-i', '--import_data', required=False, help='path to the input file', default='dataset/data_all_cont4')
    parser.add_argument('--num_layers_emb', required = False, type=int, help='determines the number of layers of embedding', default=1)
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training and test sets. a list of the form [train, test]", default="0.7:0.3")
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)
    parser.add_argument('-a','--alternate_data', required=False, type=bool, help='if true, feeds data into the net alternating between labels', default=True)
    parser.add_argument('-n','--num_auxiliary_qubits', required=False,help='number of auxiliary qubits',default=0)
    parser.add_argument('--input_type', required=False,help='indicate whether DensityMatrix or classical data is used as input',default='classical')
    
    args = parser.parse_args()
    import_name = args.import_data
    print("hyper parameters: ",args)
    alternate = args.alternate_data
    parsed_shots=args.shots
    shuffle=args.shuffle
    shuffleseed = args.shuffleseed
    n_layers_emb = args.num_layers_emb
    n_extra_qubits=int(args.num_auxiliary_qubits)

    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)
    ratio = args.partition_ratio.split(":")
    ratio = [float(entry) for entry in ratio]

    dataset = preprocessing.import_dataset(import_name,shuffle, shuffleseed)
 

    if alternate:
        dataset = preprocessing.alternate_g(dataset)
        dataset = preprocessing.get_uq_g(dataset)
    print(f"using dataset of length {len(dataset)}")
    if partition_size != 'max':
        partition_split = int(partition_size)
    else:
        partition_split=len(dataset)
    print(f'using partition size of {partition_split}')
    train_set, test_set = preprocessing.train_test(dataset, partition_split, ratio)
    print("for training:")
    preprocessing.get_info_g(train_set, True)
    print("for testing:")
    preprocessing.get_info_g(test_set, True)

    Xtrain, ytrain = preprocessing.convert_for_qiskit(train_set)
    Xtest, ytest = preprocessing.convert_for_qiskit(test_set)
    if args.input_type=='classical':
        maxData = preprocessing.get_max_data(import_name)
        minData = preprocessing.get_min_data(import_name)
        Xtrain=((Xtrain-minData)/(maxData-minData)*np.pi)
        Xtest=((Xtest-minData)/(maxData-minData)*np.pi)
        Xtrain=preprocessing.normalize_amplitude(Xtrain)
        Xtest=preprocessing.normalize_amplitude(Xtest)
        Xtrain=preprocessing.vector_to_DensityMatrix(Xtrain)
        Xtest=preprocessing.vector_to_DensityMatrix(Xtest)

#######################
### feature mapping ###
#######################

    n_inputs = int(math.log2(len(Xtrain[0].data[0])))#+n_extra_qubits
    #n_external_inputs=len(Xtrain[0].data[0])

    qc = QuantumCircuit(n_inputs)
    tmp_n_params=math.comb(n_inputs,2)     #number of gates for ising_interaction (zz) embedding, this number may change for another embedding
    n_params = (n_inputs+tmp_n_params)*n_layers_emb
    theta = ParameterVector("θ_par", n_params)

    #   
    param_index=0

    for layer in range(n_layers_emb):
        for q1 in range(n_inputs):
            for q2 in range(q1,n_inputs):
                if q1!=q2:
                    qc.rzz(theta[param_index],q1,q2)
                    param_index+=1

        for q_tmp in range(n_inputs):
            qc.ry(theta[param_index],q_tmp)
            param_index+=1

    backend=AerSimulator(method='statevector')
    quant_kernel = QuantumKernel(feature_map=qc,training_parameters=theta,quantum_instance=backend)
    loss_fun=QSVC_core.QSVC_Loss(C=1.0)
    #cb_qkt = embedding.QKTCallback()
    opt = SPSA(maxiter=100)#, callback=cb_qkt.callback)
    qk_trainer = QuantumKernelTrainer(quantum_kernel=quant_kernel,optimizer=opt, loss=loss_fun)


    qkt_results = qk_trainer.fit(Xtrain, ytrain)
    optimized_kernel = qkt_results.quantum_kernel

    mapped_train=[]
    for i in Xtrain:
        mapped_train.append(i.evolve(optimized_kernel.feature_map))
    mapped_test=[]
    for i in Xtest:
        mapped_test.append(i.evolve(optimized_kernel.feature_map))

#######################
### SVC with kernel ###
#######################

    kernel_train=QSVC_core.calc_fidelity_kernel_matrix(mapped_train,mapped_train)
    kernel_test=QSVC_core.calc_fidelity_kernel_matrix(mapped_test,mapped_train)
    svc=SVC(kernel='precomputed')
    svc.fit(kernel_train,ytrain)

    ypred=svc.predict(kernel_test)
    score=accuracy_score(ypred,ytest)
    print(score)
