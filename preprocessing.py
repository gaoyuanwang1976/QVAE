
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
    return (max(X.flatten())).round(5)

def get_min_data(filename):
    X = (np.loadtxt(filename).T[:-1]).T
    return (min(X.flatten())).round(5)

def import_dataset(filename,input_type, shuffle=False, shuffleseed=False):

    if input_type=='classical':
        array = np.loadtxt(filename)
    elif input_type=='quantum':
        f = open(filename, "r").read().splitlines()
        array=[]
        for d in f:
            array_tmp=np.zeros((len(d.split())),dtype=complex)
            for index_i,i in enumerate(d.split()):
                array_tmp[index_i]=complex(i)
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

def alternate_g(dataset):
    ones, zeros = sort_dataset(dataset)
    return coallated_dataset(ones, zeros)

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

def convert_for_qiskit_dm(dataset):
    X = []
    y = []
    for input, label in dataset:
        X.append(input)
        y.append(label)

    return(X,y)

def convert_for_qiskit_classical(dataset):
    X = []
    y = []
    for input, label in dataset:
        input = input.numpy()
        input = input.round(5)
        X.append(input)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return(X,y)


def vector_to_DensityMatrix(X):
    X_dm=[]
    for x in X:
        X_dm.append(qi.DensityMatrix(x))
    return X_dm