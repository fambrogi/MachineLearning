import itertools
import os,sys
import pandas as pd
import numpy as np

import warnings
from sdv.tabular import GaussianCopula, CopulaGAN
from sdv.tabular import CTGAN
from sdv import *

import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn import tree
from utils import make_histos_2


#from classifier import Classifier

def generateSyntheticData (ds, mode = '' , num_sample=5000):
    #I need to merge together x_train y_train
    #generate the new dataset with gaussian copula
    #split again the new data
    # fit the classifier
    # test the classifier using x_test and y_test
    # mode = 'gaussian' , ''

    print('generating artificial data for ', ds , ' dataset using ', mode, ' model ')

    x_train = pd.read_csv('splittedDatasets/x_train_' + ds + '.csv')
    y_train = pd.read_csv('splittedDatasets/y_train_' + ds + '.csv')

    if ds == 'income':
        target = 'class'
    if ds == 'titanic':
        target = 'Survived'
    if ds == 'social':
        target = 'Purchased'

    if mode == 'gaussian_copula':
        model = GaussianCopula()
    if mode == 'ctGAN' :
        model = CTGAN()
    if mode == 'copulaGAN' :
        model = CopulaGAN()


    df_all = pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit( df_all )

    synthetic_data = model.sample(num_sample)

    synthetic_data[target].to_csv('generatedData/y_train_' + ds + '_' + mode + '.csv', index=False)
    y_train_s = synthetic_data[target]

    del synthetic_data[target]
    synthetic_data.to_csv('generatedData/x_train' + ds + '_' + mode + '.csv', index=False)

    x_train_s = synthetic_data

    make_histos_2(ds, df_all, what=mode)

    return x_train_s, y_train_s



