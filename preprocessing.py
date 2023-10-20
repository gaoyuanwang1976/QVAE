
#%% Imports

import torch
from torch.autograd import Function
import torch.nn as nn
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.extensions import UnitaryGate
import numpy as np
from numpy import pi
import collections
import random
import pandas as pd
import multiprocessing as mp
from functools import partial

from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#%% YplusZ_gate definition (for use in y-measurement)

#here I write a custom defintion of a Y+Z gate because I couldn't find anything quite right in the qiskit library
yz_mtx = np.zeros((2,2), dtype=complex)
yz_mtx[0][0] = 1
yz_mtx[0][1] = -1j
yz_mtx[1][0] = 1j
yz_mtx[1][1] = -1
yz_mtx = yz_mtx*(1/np.sqrt(2)) #making sure the matrix is unitary
YplusZ_gate = UnitaryGate(yz_mtx, label="Y+Z") #casts the matrix as a qiskit circuit gate

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return np.array(v/norm)

def normalize_amplitude(X):
    X_norm=[]
    for x in X:
        X_norm.append(normalize(x))
    return np.array(X_norm)


# for data preprocessing
def get_n_qubits(dataset):
    """Extracts the number of qubits needed to learn on a given dataset"""
    n_qubits = 0
    for data in dataset:
        input, label = data
        n_qubits = len(input[0])+1
        break
    return n_qubits

def initialize(data, THRESHOLD=0.5):
    """initializes a quantum circuit for tensor of arbitrary length. Expects that data is a tensor of the form (input, label)"""
    binarized = (data[0]>THRESHOLD)
    binarized = binarized.int()
    flattened = binarized.flatten().tolist()
    qc = QuantumCircuit(len(flattened)+1, 1)
    for q in range(len(flattened)):
        if flattened[q]==1:
            qc.x(q)
    return qc

def remap(expectation):
    """Remaps expectation values in the [0, 1] range to the [-1, 1] range to make them compatible with the loss function defined in Farhi and Neven 2018"""
    return 1-2*expectation


def rescale(dataset):
    """
    Rescales a dataset such that each input's max value is 1, min value is 0, and all other values are adjusted accordingly.
    Meant to be used before binarize(). Used with MNIST data.
    Expects dataset with only positive values. Datasets with negative values should first be shifted so that the min val is zero.
    """
    newdataset = []
    for data, label in dataset:
        max = torch.max(data)
        scaled = data/max
        newdataset.append((scaled, label))
    return newdataset



#%% Function definitions specific to the MNIST problem

def get_uq_g(dataset):
    """Removes all duplicate and conflicting inputs from a dataset."""
    uq = []
    num_0 = 0
    num_1 = 0
    num_overlap = 0
    dict = collections.defaultdict(set)
    for data, label in dataset:
        key = tuple(data.flatten().tolist())
        dict[key].add(label)
    for item in dataset:
        data, label = item
        key = tuple(data.flatten().tolist())
        if dict[key] == {0}:
            num_0 +=1
            uq.append(item)
        elif dict[key] == {1}:
            num_1 +=1
            uq.append(item)
        elif dict[key] == {0,1}:
            num_overlap +=1
        else:
            print("Error with item", item)
            break
    return uq

#%% Function definitions specific to the genomics dataset

def array_to_dataset(array,input_type):
    """Converts data arrays into appropriate form for use in QNN. Assumes each entry in array is a seperate datapoint w the last column corresponding to label."""
    dataset = []
    if input_type=='classical':
        for entry in array:
            input = torch.tensor(entry[:len(entry)-1])
            label = int(entry[len(entry)-1])
            dataset.append((input,label))

    elif input_type=='quantum':
        dim=int(np.sqrt(len(array[0])-1))
        for entry in array:
            input=np.zeros((dim,dim),dtype=complex)
            for row in range(dim):
                for col in range(dim):
                    input[row][col]=entry[row*dim+col]
            label = int(entry[len(entry)-1].real) #.real to avoid warning
            dataset.append((input,label))
    return dataset

def get_max_data(filename):
    X = (np.loadtxt(filename).T[:-1]).T
    return (max(X.flatten())).round(1)

def get_min_data(filename):
    X = (np.loadtxt(filename).T[:-1]).T
    return (min(X.flatten())).round(1)

def import_dataset(filename,input_type, shuffle=False, shuffleseed=False):
    """
    Imports appropriately-formatted text matrix, converting to array then to dataset.
    Includes options to shuffle randomly or according to a given seed.
    """

    if input_type=='classical':
        array = np.loadtxt(filename)
    elif input_type=='quantum':
        f = open(filename, "r").read().splitlines()

        #dim=int(np.sqrt(len(f[0].split())-1))
        array=[]
        for d in f:
            array_tmp=np.zeros((len(d.split())),dtype=complex)
            for index_i,i in enumerate(d.split()):
                array_tmp[index_i]=complex(i)
            #array_tmp[-1]=int(d.split()[-1])
            #array_tmp=np.zeros((dim,dim),dtype=complex)
            #for row in range(dim):
            #    for col in range(dim):
            #        array_tmp[row][col]=complex(d.split()[row*dim+col])
            array.append(array_tmp)

    if shuffle:
        if shuffleseed==False:
            np.random.shuffle(array)
        else:
            np.random.seed(shuffleseed)
            np.random.shuffle(array)
    return array_to_dataset(array,input_type)

def train_test(dataset, scale, ratio):
    
    part1 = int(scale*ratio[0])
    part2 = part1 + int(scale*ratio[1])
    train = dataset[:part1]
    test = dataset[part1:part2]
    return (train, test)

def get_info_g(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        input, label = data
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1".')
    return (ones, zeros)

#genomics data comes pre-balanced, but i do have a balance_g alt function that's written in the genodock secion of qnn
# note that this is overruled by the alternate_g function, which is used by default
def balance_g(dataset):
    ones, zeros = get_info_g(dataset)
    ratio = (zeros-ones)/(zeros)
    balanced = []
    for item in dataset:
        data, label = item
        if label == 0:
            if random.random()>ratio:
                balanced.append(item)
        else:
            balanced.append(item)
    return balanced

def alternate_g(dataset):
    ones, zeros = sort_dataset(dataset)
    return coallated_dataset(ones, zeros)

#note sort_df below is more general. keeping both for convenience
def sort_genodock(df):
    "used in data preprocessing"
    df1 = df[df['BA_Change_Vina']==0]
    df2 = df[df['BA_Change_Vina']==1]
    return (df1, df2)

def sort_df(df, labelcolname):
    "used in data preprocessing. splits dataset according to label"
    df1 = df[df[labelcolname]==0]
    df2 = df[df[labelcolname]==1]
    return (df1,df2)

def alternate_df(df1, df2):
    if len(df1) < len(df2):
        length = len(df1)
        new_df = df1.copy()
    else:
        length = len(df2)
        new_df = df2.copy()
    for i in range(length):
        new_df.loc[i] = df1.iloc[i] #changing to iloc from loc
        new_df.loc[i+1]= df2.iloc[i] #changing to iloc from loc

    return new_df

def sort_dataset(dataset, a_label=1):
    labeled_a = []
    labeled_b = []
    for data in dataset:
        input, label = data
        if label == a_label:
            labeled_a.append(data)
        else:
            labeled_b.append(data)
    return (labeled_a, labeled_b)

def coallated_dataset(set1, set2):
    dataset = []
    if len(set1)<len(set2):
        length = len(set1)
    else:
        length = len(set2)
    for i in range(length):
        dataset.append(set1[i])
        dataset.append(set2[i])
    return dataset

def import_all_coallated(shuffle=False, shuffleseed=False):
    ds = import_dataset('genomics_datasets','data_all')
    if shuffle:
        if shuffleseed == False:
            random.shuffle(ds)
        else:
            random.Random(shuffleseed).shuffle(ds)
    ones, zeros = sort_dataset(ds)
    return coallated_dataset(ones, zeros)

def get_ytrue_yscore(model, params, dataset, binarize=False, cont=True):
    ytrue = []
    yscore = []
    for data in dataset:
        input, label = data
        ytrue.append(label.item())
        if binarize:
            yscore.append(round(model.roc_forward(params,cont, data)))
        else:
            yscore.append(model.roc_forward(params, cont, data))
    ytrue = np.asarray(ytrue)
    yscore = np.asarray(yscore)
    return (ytrue,yscore)

def round_yscore(yscore):
    rounded = []
    for item in yscore:
        rounded.append(round(item))
    return rounded


# %% defs for qisQNN

def convert_for_qiskit_dm(dataset):
    X = []
    y = []
    for input, label in dataset:
        #input = input.numpy()
        #input = input.round(1)
        X.append(input)
        y.append(label)

    return(X,y)

def convert_for_qiskit_classical(dataset):
    X = []
    y = []
    for input, label in dataset:
        input = input.numpy()
        input = input.round(1)
        X.append(input)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return(X,y)

def remove_feature(featnum, num_layers, params):
    """
    writing a function to adjust models for smaller feature sets by removing all parameters corresponding to a given feature.
    features are numbered starting from 0, and whichever number is given to the function will be removed from the parameter set.
    """
    num_features = len(params)/num_layers
    newparams = []
    for i in range(len(params)):
        if i%num_features == featnum:
            pass
        else:
            newparams.append(params[i])
    return newparams

# %% import data def
def import_data(parsed_dataset, parsed_batch_size, partition_size, genomics_dataset, genodock_set, alternate, continuous, shuffle, converttype, ratio = [1,1,1]):
    if parsed_dataset=="genomics":
        if genomics_dataset == "data_all":
            dataset = import_all_coallated(shuffle)
        else:
            dataset = import_dataset('genomics_datasets',genomics_dataset, shuffle)
        if alternate:
            dataset = alternate_g(dataset)
        train_set, val_set, test_set = train_val_test(dataset, partition_size, ratio)
        Xtrain, ytrain = convert_for_qiskit(train_set, converttype)
        Xval, yval = convert_for_qiskit(val_set, converttype)
        Xtest, ytest = convert_for_qiskit(test_set, converttype)
        get_info_g(train_set, True)
        print("for the validation:")
        get_info_g(val_set, True)
        print("for testing:")
        get_info_g(test_set, True)
        val_set = torch.utils.data.DataLoader(val_set, shuffle=False) #any necessary shuffling happens before this
        test_set = torch.utils.data.DataLoader(test_set, shuffle=False)

    elif parsed_dataset=="genodock":
        dataset = import_dataset('genodock_preprocessed','genodock_'+str(genodock_set)+".txt", shuffle)
        if alternate:
            dataset = alternate_g(dataset)
        train_set, val_set, test_set = train_val_test(dataset, partition_size, ratio)
        Xtrain, ytrain = convert_for_qiskit(train_set, converttype)
        Xval, yval = convert_for_qiskit(val_set, converttype)
        Xtest, ytest = convert_for_qiskit(test_set, converttype)
        get_info_g(train_set, True)
        print("for the validation:")
        get_info_g(val_set, True)
        print("for testing:")
        get_info_g(test_set, True)
        test_set = torch.utils.data.DataLoader(test_set, shuffle=False)

    else:
        print("error with dataset hyperparameter!")
    return (train_set,val_set,test_set,Xtrain,ytrain,Xval,yval,Xtest,ytest)

#%% ROC and saliency

def get_roc_df(models, test_set, Xtest, ytest,parsed_shots, parsed_shift, n_qubits, num_layers=3, full_cross=False, layerorder='zxzxzx'):
    """
    Expects a list of models, each of which is a touple of the form
    (params [list if qnn, full model if nn], model_name [str], modeltype ['nn' or 'qnn']).
    Returns an AUC list and a DataFrame which can be use to plot ROC curves.
    """
    namelist = []
    fprlist = []
    tprlist = []
    AUClist = []

    for params, name, modeltype in models:
        if modeltype == 'qnn':
            if '6' in name:
                num_layers = 6
            model = Net(parsed_shots, parsed_shift, n_qubits, num_layers, layerorder)
            model.eval()
            ytrue, yscore = get_ytrue_yscore(model, params, test_set)
            if 'qisQNN' in name: # prediction flip hack that I still need to get to the root of !!
                yscore = 1-yscore

        elif modeltype.lower() == 'crossqnn':
            if "full" in name.lower():
                full_cross = True
            model = Net_cross(parsed_shots, parsed_shift, n_qubits, num_layers, full_cross)
            model.eval()
            ytrue, yscore = get_ytrue_yscore(model, params, test_set)

        elif modeltype == 'logreg':
            yscore = params.decision_function(Xtest)
            ytrue = ytest
        else:
            yscore = [item[1] for item in params.predict_proba(Xtest)]
            ytrue = ytest

        print(f'for {name}, the confusion matrix is: {confusion_matrix(ytrue, round_yscore(yscore)).ravel()}')
        fpr1, tpr1, thresholds = roc_curve(ytrue, yscore)
        fprlist += list(fpr1)
        tprlist += list(tpr1)

        auc1 = auc(fpr1, tpr1)
        AUClist.append((auc1, name))
        namelist += [name+f' (AUC = {round(auc1,2)})']*len(fpr1)

    roc_dict = {"model_name":namelist,
               "fpr":fprlist,
               "tpr":tprlist}

    return (AUClist, pd.DataFrame.from_dict(roc_dict))

def get_shift_loss(data, model, params, shift, modeltype, continuous=True):
    """returns a list of the unshifted loss followed the loss when each element is shifted independently"""
    input, label = data
    losses = []
    losses.append(model.eval_forward(params, continuous, data))
    inputlist = input[0].tolist()
    for i in range(len(inputlist)):
        shiftedlist = inputlist.copy()
        shiftedlist[i] -= shift
        shiftedinput = torch.Tensor(shiftedlist)
        shifteddata = (shiftedinput, label)
        shift_loss = model.eval_forward(params, continuous, shifteddata)
        losses.append(shift_loss)
    return losses




def nn_loss(X, y, datanum, model):
    pred = model.predict_proba(X[datanum:datanum+1])[0][1]
    return 1-(remap(y[datanum])*remap(pred))

def get_shift_loss_nn(X, y, datanum, model, shift):
    losses = []
    losses.append(nn_loss(X,y, datanum, model))
    datum = X[datanum]
    for i in range(len(datum)):
        Xcopy = X.copy()
        shifteddatum = datum.copy()
        shifteddatum[i] -= shift
        Xcopy[datanum] = shifteddatum
        shift_loss = nn_loss(Xcopy, y, datanum, model)
        losses.append(shift_loss)
    return losses

def get_saliency_nn(X, y, n_qubits, model, modelname, shift, maxdata,signed=False,convert=True):
    columns = []
    columns.append('datanum')
    columns.append("modelname")
    for n in range(n_qubits-1):
        columns.append("feature_"+str(n))
    df = pd.DataFrame(columns = columns)
    for datanum in range(len(X)):
        if datanum > maxdata:
            break
        losses = get_shift_loss_nn(X, y, datanum, model, shift)
        baseloss = losses[0]
        shiftlosses = losses[1:]
        saliencylist = []
        for item in shiftlosses:
            saliency = (baseloss-item)/shift
            # adding the conversion
            label = y[datanum]
            label = remap(label)
            if convert:
                saliency = saliency*(-1/label)
            saliencylist.append(saliency)
        paramsetcounter = 0
        if signed:
            df.loc[len(df)]=[paramsetcounter, str(modelname)]+saliencylist
        else:
            abslist = [abs(num) for num in saliencylist]
            df.loc[len(df)]=[paramsetcounter, str(modelname)]+abslist
    df['datanum'] = df['datanum'].astype(int)

    return df

def saliency_plot(saliency_df, title='Saliency comparison between models, averaged across data inputs', rescale=True):
    sliced = saliency_df[['modelname']+['feature_'+str(i) for i in range(len(saliency_df.columns)-2)]]
    avgd = sliced.groupby(['modelname']).mean()
    if rescale:
        avgd = avgd.div(avgd.abs().max(axis=1), axis=0)
    fig = avgd.plot.bar(title=title)
    fig.legend(bbox_to_anchor=(1.0, 1.0))



# %% logreg

def prep_plot_components_lr(X, y, X2, y2):
    ###the logistic regression###
    scores = []
    for i in range(10):
        clf = LogisticRegression(C=10**(1-i), penalty='l2', solver='liblinear')
        sc = cross_val_score(clf, X, y, cv=5)
        scores.append(sc.mean())

    scores_df = pd.DataFrame({'i': range(1, len(scores) + 1), 'score': scores})

    i_star = scores_df['score'].idxmax()

    clf_lr = LogisticRegression(C=10**(1-i_star), penalty='l2', solver='liblinear')


    clf_lr.fit(X, y)

    print(f'coefficients: {clf_lr.coef_}')
    print(f'intercept: {clf_lr.intercept_}')
    ### - - - - ###

    #getting fpr and tpr
    y_score = clf_lr.decision_function(X2)
    fpr, tpr, _ = roc_curve(y2, y_score)

    return (fpr, tpr, auc(fpr,tpr))

def split_labels(df):
    labelcol = len(df.columns)-1
    X = df.drop(str(labelcol), axis=1)
    y = df[str(labelcol)]
    return(X,y)

def get_logreg(X,y):
    ###the logistic regression###
    scores = []
    for i in range(10):
        clf = LogisticRegression(C=10**(1-i), penalty='l2', solver='liblinear')
        sc = cross_val_score(clf, X, y, cv=5)
        scores.append(sc.mean())

    scores_df = pd.DataFrame({'i': range(1, len(scores) + 1), 'score': scores})

    i_star = scores_df['score'].idxmax()

    clf_lr = LogisticRegression(C=10**(1-i_star), penalty='l2', solver='liblinear')


    clf_lr.fit(X, y)

    print(clf_lr.coef_)
    print(clf_lr.intercept_)
    ### - - - - ###
    return clf_lr

#%% param saliency defs

def qnn_params_shift_loss(data, model, params, shift, continuous=True):
    """returns a list of the unshifted loss followed the loss when each parameter is shifted independently (for qnn)"""
    input, label = data
    losses = []
    losses.append(model.eval_forward(params, continuous, data))
    for i in range(len(params)):
        shiftedparams = params.copy()
        shiftedparams[i] -= shift
        shift_loss = model.eval_forward(shiftedparams, continuous, data)
        losses.append(shift_loss)
    return losses

def get_param_saliency(dataset, n_qubits, modeltuple, shift, maxdata, parsed_shots, parsed_shift, num_layers, layerorder, full_cross, signed=False, convert=True):
    """
    Returns a dataframe which describes input saliencies across various (trained) QNN models. Arguments are as follows:
        'dataset': The data for which saliency is desired. Data is expected in the form (input, label), and dataset is expected to be a list of data tuples.
        'n_qubits': Number of qubits in the QNN, i.e. number of input values + 1
        'modelset': A list of final models for which the saliency is wanted. Each entry is a tuple with (model, modeltype, thetalist)
        'shift': Amount by which parameters should be shifted.
        'maxdata': Caps the number of datapoints for which the saliency is computed
    """

    params, modelname, modeltype = modeltuple
    if 'qnn' not in modeltype.lower(): #only meant to work with qnns
        pass
    else:
        if 'cross' in modeltype.lower():
            model = Net_cross(parsed_shots, parsed_shift, n_qubits, num_layers, full_cross)
            model.eval()
            all_sal = []
            datacounter = 0
            for data in dataset:
                if datacounter > maxdata:
                        break
                losses = qnn_params_shift_loss(data, model, params, shift, continuous=True)
                baseloss = losses[0]
                sal = []
                for item in losses[1:]:
                    saliency = (baseloss-item)/shift
                    input, label = data
                    label = remap(label.item())
                    if convert:
                        saliency = saliency*(-1/label)
                    if signed:
                        sal.append(saliency)
                    else:
                        sal.append(abs(saliency))
                all_sal.append(sal)
                datacounter+=1
            df = pd.DataFrame(all_sal)
            return df
        else:
            model = Net(parsed_shots, parsed_shift, n_qubits, num_layers, layerorder)
            model.eval()
            all_sal = []
            datacounter = 0
            for data in dataset:
                if datacounter > maxdata:
                        break
                losses = qnn_params_shift_loss(data, model, params, shift, continuous=True)
                baseloss = losses[0]
                sal = []
                for item in losses[1:]:
                    saliency = (baseloss-item)/shift
                    input, label = data
                    label = remap(label.item())
                    if convert:
                        saliency = saliency*(-1/label)
                    if signed:
                        sal.append(saliency)
                    else:
                        sal.append(abs(saliency))
                all_sal.append(sal)
                datacounter+=1
            df = pd.DataFrame(all_sal)
            return df

def get_readout(x):
    """
    given an output string, returns only the readout qubit. Since qiskit orders qubits in reverse order of importance,
    the readout qubit will be the first number of the output string.
    """
    return(int(str(x)[0]))


def vector_to_DensityMatrix(X):
    X_dm=[]
    for x in X:
        X_dm.append(qi.DensityMatrix(x))
    return X_dm