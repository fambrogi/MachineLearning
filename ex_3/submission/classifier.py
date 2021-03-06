import itertools
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter('ignore')

from sdv.tabular import GaussianCopula, CopulaGAN
from sdv.tabular import CTGAN
from sdv import *

import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier

from generate_data import generateSyntheticData
from clean_datasets import dic  # dic containing all the info about data frames
from utils import *


""" Define the output directory for Plots (global) """
plot_dir = 'Plots'
if not os.path.isdir(plot_dir):
    os.system('mkdir ' + plot_dir )




""" Dictionary containing the scores """

res_all = { 'income': { 'precision': [],
                        'accuracy': [],
                        'recall': [] },
            'titanic': { 'precision': [],
                        'accuracy': [],
                        'recall': [] },

            'social': { 'precision': [],
                        'accuracy': [],
                        'recall': [] },

            }



def splitDataset(df='', ds='' ):
    """ Split the data frame df provided into train and test.
        ds: name of the data set
        df: data frame """

    train_set = dic[ds]['train_features']
    target_set = dic[ds]['target_features']

    train_set = df[train_set]
    target_set = df[target_set]

    x_train, x_test, y_train, y_test = train_test_split(train_set, target_set,
                                                        test_size=0.30,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test




def predict(x_test, classifier):
    """ Wraper for the sklearn predict function """
    y_pred=classifier.predict(x_test)
    return y_pred



def trainEvaluateData(x_train, x_test, y_train, y_test, ds = '', what='', classifier='forest', norm_confusion = 'true'):
    """ Train and evaluate a model """
    
    if classifier == 'forest':
        cl = RandomForestClassifier();

    cl.fit(x_train, y_train)

    y_pred = cl.predict(x_test)

    confusion_m = confusion_matrix(y_test, y_pred, normalize= norm_confusion)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    res_all[ds]['accuracy'].append(accuracy)
    res_all[ds]['precision'].append(report['macro avg']['precision'])
    res_all[ds]['recall'].append(report['macro avg']['recall'])


    printConfusionMatrix(confusion_m, ds , title ='', what=what)



def splitOriginalData(ds, rows=None, balance = True):
    """ ds: data set name,
        rows: if set, gives the total number of rows to consider (for testing / faster processing )
        balance: [True, False]  if True, balances the majority target class between the two possible outcome
        """
    print('classification of ' , ds , ' dataset')

    if not os.path.isdir('splittedDatasets'):
        os.system('mkdir splittedDatasets')

    # reading cleaned dataset
    dataset = pd.read_csv("input_data/" + ds + '_cleaned.csv')

    # balancing the data sets to have 50% equal target class
    if balance:
        values, counts = np.unique( dataset[dic[ds]['target_features'][0]], return_counts = True )
        df_f = dataset.loc [ dataset[ dic[ds]['target_features'][0] ] == min(values) ]
        df_s = dataset.loc [ dataset[ dic[ds]['target_features'][0] ] == max(values) ]

        if len(df_f) > len(df_s):
            df_down = df_f.sample( len(df_s) )
            dataset = pd.concat( [df_down, df_s ])

        elif len(df_s) > len(df_f) :
            df_down = df_s.sample( len(df_f) )
            dataset = pd.concat( [df_down, df_f ])

    # extracting a smaller set of rows for faster testing
    if rows:
        n = min(rows, len(dataset))
        dataset = dataset.sample(n)

    # split the datasets
    x_train, x_test, y_train, y_test = splitDataset(df=dataset, ds=ds)

    x_train.to_csv('splittedDatasets/x_train_' + ds + '.csv', index=False)
    x_test.to_csv('splittedDatasets/x_test_' + ds + '.csv', index=False)

    y_train.to_csv('splittedDatasets/y_train_' + ds + '.csv', index=False)
    y_test.to_csv('splittedDatasets/y_test_' + ds + '.csv', index=False)

    return x_train, x_test, y_train, y_test



modes = ['gaussian_copula', 'ctGAN', 'copulaGAN'] # available synthetic data methods
datasets = ['titanic', 'social', 'income'] # available data sets


""" Main """

# select rows = 40000 to process the entire data sets available (slow!)
def main():

    for ds in datasets:

        x_train, x_test, y_train, y_test = splitOriginalData(ds, rows=100, balance = True )
        d = trainEvaluateData(x_train, x_test, y_train, y_test, ds=ds, what='Original', classifier='forest', norm_confusion='true')

        for mode in modes:
            x_train_s, y_train_s = generateSyntheticData(ds, mode=mode, num_sample=len(x_train))
            d = trainEvaluateData(x_train_s, x_test, y_train_s, y_test, ds=ds, what='Syntethic_' + mode, classifier='forest', norm_confusion='true' )

        # Plotting summary information
        plot_histo_comparison_ds(ds, columns = dic[ds]['train_features'])
        print(res_all)
        plot_results(res_all, ds)




if __name__=="__main__":
    main()




