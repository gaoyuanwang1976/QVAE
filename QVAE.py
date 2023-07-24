import numpy as np
#import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM

from qiskit_machine_learning.neural_networks import CircuitQNN,SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor

from qiskit.utils import algorithm_globals
algorithm_globals.random_seed = 42
import math
import core
import preprocessing
from math import comb
import time as time
import copy

import os
abspath = os.path.abspath('__file__')
dname = os.path.dirname(abspath)
os.chdir(dname)

import argparse

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=10)
    parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=5)
    parser.add_argument('-i', '--itl_params', required=False, type=float, help='for nonrandom initial parameters, sets every param to this value', default=np.pi/2)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of alternating R_ZX and R_XX layers in the QNN', default=1)
    parser.add_argument('-g', '--genomics_dataset', required=False, help='determines which number genomics dataset to run. defaults to dataset 1', default=1)
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train, test, and validation sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
    parser.add_argument('-o','--optimizer', required=False, type=str, help='determines the Qiskit optimizer used in qnn', default='cobyla')
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)


    args = parser.parse_args()

    import_name = 'dataset/data_1'

    alternate = True
    parsed_shots=args.shots
    shuffle=args.shuffle
    shuffleseed = args.shuffleseed
    n_layers = args.num_layers

    partition_size=args.partition_size
    if partition_size != 'max':
        parition_size = int(partition_size)
    ratio = args.partition_ratio.split(":")
    ratio = [float(entry) for entry in ratio]

    num_epoch=args.epochs

    if args.optimizer.lower() == 'cobyla':
        optimizer = COBYLA
    elif args.optimizer.lower() == 'spsa':
        optimizer = SPSA
    elif args.optimizer.lower() == 'adam':
        optimizer = ADAM
    else:
        print("problem with parsing optimizer, defaulting to COBYLA")
        optimizer = COBYLA

    dataset = preprocessing.import_dataset(import_name,shuffle, shuffleseed)
    maxData = preprocessing.get_max_data(import_name)
    minData = preprocessing.get_min_data(import_name)

    if alternate:
        dataset = preprocessing.alternate_g(dataset)
        dataset = preprocessing.get_uq_g(dataset)
    print(f"using dataset of length {len(dataset)}")
    if partition_size != 'max':
        partition_split = int(partition_size)
    else:
        partition_split=len(dataset)
    print(f'using partition size of {partition_split}')
    train_set, val_set, test_set = preprocessing.train_val_test(dataset, partition_split, ratio)
    train_len = len(train_set)
    preprocessing.get_info_g(train_set, True)
    print("for testing:")
    preprocessing.get_info_g(test_set, True)
    test_len = len(test_set)

    Xtrain, ytrain = preprocessing.convert_for_qiskit(train_set)
    Xval, yval = preprocessing.convert_for_qiskit(val_set)
    Xtest, ytest = preprocessing.convert_for_qiskit(test_set)
    n_dim=4
    Xtrain=((Xtrain-minData)/(maxData-minData)*np.pi).T[:n_dim].T
    Xval=((Xval-minData)/(maxData-minData)*np.pi).T[:n_dim].T
    Xtest=((Xtest-minData)/(maxData-minData)*np.pi).T[:n_dim].T
    Xtrain=preprocessing.normalize_amplitude(Xtrain)
    Xval=preprocessing.normalize_amplitude(Xval)
    Xtest=preprocessing.normalize_amplitude(Xtest)



#######################
##### circuit def #####
#######################

    n_features = len(Xtrain[0])
    n_qubit = math.log2(n_features)
    assert(int(n_qubit)==n_qubit)
    n_qubit=int(n_qubit)
    x_params = ParameterVector('x',n_features)
    tmp_gates=comb(n_qubit,2)     #number of gates for ising_interaction (zz) embedding, this number may change for another embedding
    n_gates = (n_qubit+tmp_gates)*n_layers         
    theta_params = ParameterVector('theta', 2*n_gates)

    num_encoder_params=n_gates

    qc_e=core.encoder(n_layers,n_qubit,theta_params[:num_encoder_params])
    qc_d=core.decoder(n_layers,n_qubit,theta_params[num_encoder_params:])

    qnn = core.QVAE_NN(circuit=qc_e, encoder=qc_e,decoder=qc_d,input_params=x_params, weight_params=theta_params,num_encoder_params=num_encoder_params)
    #qnn = SamplerQNN(circuit=qc_e, input_params=x_params, weight_params=theta_params_e)

    #qnn_weights = algorithm_globals.random.random(qnn.num_weights)

    #model=NeuralNetworkRegressor(neural_network=qnn,optimizer=optimizer(),loss= 'squared_error',warm_start=True,initial_point=qnn_weights)
    model=core.QVAE_trainer(neural_network=qnn,optimizer=optimizer(),loss= 'squared_error',warm_start=True)

    best_val_score=0

    for epoch in range(num_epoch):
        model.fit(Xtrain, Xtrain)
        this_train_score=model.score(Xtrain, Xtrain)
        this_val_score=model.score(Xval, Xval)
        print(epoch,this_train_score,this_val_score)

        if this_val_score > best_val_score: #validation wrapper
            best_val_epoch = epoch
            best_val_acc=this_val_score
            best_model= copy.deepcopy(model)
            patience_counter = 0
            print(f"new best validation score {best_val_acc}")
        else:
            patience_counter+=1
        if patience_counter == args.patience:
            print("ran out of patience")
            break

    trainscore = best_model.score(Xtrain, Xtrain)
    testscore = best_model.score(Xtest, Xtest)
    valscore = best_model.score(Xval, Xval)
    print(f'best model train score: {trainscore}')
    print(f'best model test score: {testscore}')
    print(f'best model val score: {valscore}')
  