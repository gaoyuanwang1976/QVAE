from pyexpat.errors import XML_ERROR_RESERVED_PREFIX_XML
from re import A
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM

import qiskit.quantum_info as qi
from qiskit.utils import algorithm_globals
#algorithm_globals.random_seed = 0
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
import sys
import argparse

#for performance metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

# %% parsing

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Simulate a QNN with the appropriate hyperparameters.")
    parser.add_argument('-e','--epochs', required=False, type=int, help='the desired number of epochs to run', default=10)
    parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=5)
    parser.add_argument('--num_layers', required = False, type=int, help='determines the number of alternating layers in the circuit', default=1)
    parser.add_argument('-i', '--import_data', required=False, help='path to the input file', default='dataset/data_all_cont4')
    parser.add_argument('--partition_size', required=False, help='sets partition size for splitting data into train and test sets (scales the partition_ratio arg)', default='max')
    parser.add_argument('--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training and test sets. a list of the form [train, test]", default="0.7:0.3")
    parser.add_argument('-o','--optimizer', required=False, type=str, help='determines the Qiskit optimizer used in qnn', default='cobyla')
    parser.add_argument('-x','--shots', required=False, type=int, help="the number of shots per circuit simulation", default=100)
    parser.add_argument('--shuffle', required=False, type=bool, help='determines whether to shuffle data before alternating', default=False)
    parser.add_argument('--shuffleseed', required=False, type=int, help='a seed for use in shuffling the dataset, if left False and --shuffle=True, will be completely random', default=False)
    
    parser.add_argument('-t','--num_trash_qubits', required=False, type=int, help='number of trash qubits, the first 0,...,N-1 qubits will be traced out in the latent space', default=1)
    parser.add_argument('--input_dim', required=False, type=int, help='customize the input data dimension, if zero the original input dimension is preserved', default=0)
    parser.add_argument('--reconstruction_loss', required=False, type=str, help='define the loss used in the reconstruction term of the objective', default='fidelity')
    parser.add_argument('--beta_weight', required=False, type=float, help='the beta parameter that controlls the relative weight of the quantum entropy term in the objective', default=1.0)
    parser.add_argument('--regularizer_type', required=False, type=str, help='choose between KL-Divergence and JS-Divergence', default='JSD')
    parser.add_argument('--num_auxiliary_encoder', required=False, type=int, help='number of auxiliary qubits in the encoder', default=0)
    parser.add_argument('--num_auxiliary_decoder', required=False, type=int, help='number of auxiliary qubits in the decoder', default=0)
    parser.add_argument('--global_state', action='store_true', help='whether a global quantum state is used')

    parser.add_argument('--output_dir',required=False, help='output directory for reconstructed state and latent state',default=None)



    args = parser.parse_args()

    import_name = args.import_data
    num_epoch=args.epochs
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
    
    ######### QVAE specific parameters
    trash_qubits=list(range(args.num_trash_qubits))

    reconstruction_loss=args.reconstruction_loss
    if reconstruction_loss not in ['cross_entropy','fidelity','wasserstein','JSD']:
        reconstruction_loss='fidelity'
        print('reconstruction loss choice not recognized, using fidelity loss')

    beta_weight=args.beta_weight

    regularizer_type=args.regularizer_type
    if regularizer_type not in ['KLD','JSD','fidelity']:
        print('divergence type not recognized, use JSD instead')
        regularizer_type='JSD'

    num_auxiliary_encoder=args.num_auxiliary_encoder
    num_auxiliary_decoder=args.num_auxiliary_decoder

    ##########

    if args.optimizer.lower() == 'cobyla':
        optimizer = COBYLA
    elif args.optimizer.lower() == 'spsa':
        optimizer = SPSA
    elif args.optimizer.lower() == 'adam':
        optimizer = ADAM
    else:
        print("problem with parsing optimizer, defaulting to COBYLA")
        optimizer = COBYLA

    dataset = preprocessing.import_dataset(import_name,'classical',shuffle, shuffleseed)
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
    train_set, test_set = preprocessing.train_test(dataset, partition_split, ratio)
    train_len = len(train_set)
    print("for training:")
    preprocessing.get_info_g(train_set, True)
    print("for testing:")
    preprocessing.get_info_g(test_set, True)
    test_len = len(test_set)

    Xtrain, ytrain = preprocessing.convert_for_qiskit_classical(train_set)
    Xtest, ytest = preprocessing.convert_for_qiskit_classical(test_set)
    if args.input_dim==0:
        n_dim=len(Xtrain[0])
    else:
        n_dim=args.input_dim
    Xtrain=((Xtrain-minData)/(maxData-minData)*np.pi).T[:n_dim].T
    Xtest=((Xtest-minData)/(maxData-minData)*np.pi).T[:n_dim].T
    Xtrain=preprocessing.normalize_amplitude(Xtrain)
    Xtest=preprocessing.normalize_amplitude(Xtest)     

#######################
##### circuit def #####
#######################

    n_features = len(Xtrain[0])
    n_qubit = math.log2(n_features)
    assert(int(n_qubit)==n_qubit)
    n_qubit=int(n_qubit)
    x_params = ParameterVector('x',n_features)
    
    # for encoder
    n_qubit_e=n_qubit+num_auxiliary_encoder

    tmp_gates_e=comb(n_qubit_e,2)     #number of gates for ising_interaction (zz) embedding, this number may change for another embedding
    n_gates_e = (n_qubit_e+tmp_gates_e)*n_layers         

    # for decoder
    n_qubit_d=n_qubit+num_auxiliary_decoder
    tmp_gates_d=comb(n_qubit_d,2)
    n_gates_d = (n_qubit_d+tmp_gates_d)*n_layers

    theta_params = ParameterVector('theta', n_gates_e+n_gates_d)

    num_encoder_params=n_gates_e

    qc_e=core.encoder(n_layers,n_qubit,theta_params[:num_encoder_params],num_auxiliary_encoder)
    qc_d=core.decoder(n_layers,n_qubit,theta_params[num_encoder_params:],num_auxiliary_decoder)

    qnn = core.QVAE_NN(circuit=qc_e, encoder=qc_e,decoder=qc_d,input_params=x_params, weight_params=theta_params,num_encoder_params=num_encoder_params,trash_qubits=trash_qubits,num_auxiliary_encoder=num_auxiliary_encoder,num_auxiliary_decoder=num_auxiliary_decoder)
    #qnn = SamplerQNN(circuit=qc_e, input_params=x_params, weight_params=theta_params_e)

    #qnn_weights = algorithm_globals.random.random(qnn.num_weights)
    #model=NeuralNetworkRegressor(neural_network=qnn,optimizer=optimizer(),loss= 'squared_error',warm_start=True,initial_point=qnn_weights)
    model=core.QVAE_trainer(neural_network=qnn,optimizer=optimizer(maxiter=100),loss= 'squared_error',warm_start=True,reconstruction_loss=reconstruction_loss,beta=beta_weight,regularizer_type=regularizer_type)

    ### convert input into density matrices ###
    Xtrain_dm=[]
    Xtest_dm=[]

    if args.global_state == True:
        combined_state_tr=np.zeros((len(Xtrain[0]),len(Xtrain[0])))
        combined_state_te=np.zeros((len(Xtest[0]),len(Xtest[0])))

    for x in Xtrain:
        Xtrain_dm.append(qi.DensityMatrix(x).data)
        if args.global_state == True:
            combined_state_tr=combined_state_tr+Xtrain_dm[-1]*1./len(Xtrain)

    for x in Xtest:
        Xtest_dm.append(qi.DensityMatrix(x).data)
        if args.global_state == True:
            combined_state_te=combined_state_te+Xtest_dm[-1]*1./len(Xtest)

    if args.global_state == True:
        if reconstruction_loss=='wasserstein':
            Xtrain_pool=[]
            Xtest_pool=[]
            tr_eigenvalues, tr_eigenvectors = np.linalg.eig(combined_state_tr)
            for tr_i,tr_eigenvalue in enumerate(tr_eigenvalues):
                count=int(round(tr_eigenvalue.real,3)*1000)
                for i in range(count):
                    Xtrain_pool.append(qi.DensityMatrix(tr_eigenvectors[tr_i]).data)

            te_eigenvalues, te_eigenvectors = np.linalg.eig(combined_state_te)
            for te_i,te_eigenvalue in enumerate(te_eigenvalues):
                count=int(round(te_eigenvalue.real,3)*1000)
                for i in range(count):
                    Xtest_pool.append(qi.DensityMatrix(te_eigenvectors[te_i]).data)

        else:
            Xtrain=np.array([combined_state_tr])
            Xtest=np.array([combined_state_te])

    else:
        Xtrain=np.array(Xtrain_dm)
        Xtest=np.array(Xtest_dm)

    best_train_score=-sys.maxsize

    for epoch in range(num_epoch):
        print('start fitting')
        model.fit(Xtrain, Xtrain)

        this_train_score=model.score(Xtrain, Xtrain,reconstruction_loss)

        this_train_fidelity=model.score(Xtrain, Xtrain,'fidelity')

        print(epoch,'train fidelity:',this_train_fidelity,'train score:',this_train_score)

        if this_train_score > best_train_score: #validation wrapper
            best_train_epoch = epoch
            best_train_score=this_train_score
            best_model= copy.deepcopy(model)
            patience_counter = 0
            print(f"new best train score {best_train_score}")
        else:
            patience_counter+=1
        if patience_counter == args.patience:
            print("ran out of patience")
            break

    trainscore = best_model.score(Xtrain, Xtrain,'fidelity')
    testscore = best_model.score(Xtest, Xtest,'fidelity')
    print(f'best model train score: {trainscore}')
    print(f'best model test score: {testscore}')

    if args.output_dir!=None:
        output_test_tmp,latent_test_tmp=best_model.predict(Xtest)

        output_train_tmp,latent_train_tmp=best_model.predict(Xtrain)
        output_train=[]
        latent_train=[]
        output_test=[]
        latent_test=[]
        for train_o,train_l,y in zip(output_train_tmp,latent_train_tmp,ytrain):
            output_train.append(np.append(train_o.data.flatten(),y))
            latent_train.append(np.append(train_l.data.flatten(),y))

        for test_o,test_l,y in zip(output_test_tmp,latent_test_tmp,ytest):
            output_test.append(np.append(test_o.data.flatten(),y))
            latent_test.append(np.append(test_l.data.flatten(),y))
        os.mkdir('./'+args.output_dir)
        np.savetxt(args.output_dir+'/reconstruct_train.dat', output_train)
        np.savetxt(args.output_dir+'/latent_train.dat', latent_train)
        np.savetxt(args.output_dir+'/reconstruct_test.dat', output_test)
        np.savetxt(args.output_dir+'/latent_test.dat', latent_test)


