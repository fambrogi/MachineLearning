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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree



from generate_data import generateSyntheticData
from clean_datasets import dic  # dic containing all the info about data frames
from utils import * # printConfusionMatrix


""" Define the output directory for Plots (global) """
plot_dir = 'Plots'
if not os.path.isdir(plot_dir):
    os.system('mkdir ' + plot_dir )




""" Dictionary containing the training and prediction features """
features = { 'adult' :
                 {'features':  ["age","workclass","fnlwgt","education","education-num",
                                "marital-status","occupation","relationship","race","sex",
                                "capital-gain","capital-loss","hours-per-week","native-country"],
                  'target': ["class"] }          }






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

    return train_set, target_set, x_train, x_test, y_train, y_test




def predict(x_test, classifier):
    """ Wraper for the sklearn predict function """
    y_pred=classifier.predict(x_test)
    return y_pred


"""
def evaluation(y_test,y_pred, normalize='true'):
    confusion_m = confusion_matrix(y_test, y_pred, normalize=normalize)
    accuracy = accuracy_score(y_test,y_pred)

    # return the class_report as a dictionary
    report = classification_report(y_test, y_pred, output_dict= True)
    return confusion_m, accuracy, report
"""

def trainEvaluateData(x_train, x_test, y_train, y_test, ds = '', what='', classifier='forest', norm_confusion = 'true'):

    if classifier == 'forest':
        cl = RandomForestClassifier();

    cl.fit(x_train, y_train)

    y_pred = predict(x_test, cl, )

    confusion_m = confusion_matrix(y_test, y_pred, normalize= norm_confusion)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])
    printConfusionMatrix(confusion_m, ds , title ='', what=what)



def splitOriginalData(ds):
    print('classification of ' , ds , ' dataset')

    # reading cleaned dataset
    dataset = pd.read_csv("input_data/" + ds + '_cleaned.csv')

    # split the datasets
    x, y, x_train, x_test, y_train, y_test = splitDataset(df=dataset, ds=ds)

    x_train.to_csv('splittedDatasets/x_train_' + ds + '.csv', index=False)
    x_test.to_csv('splittedDatasets/x_test_' + ds + '.csv', index=False)

    y_train.to_csv('splittedDatasets/y_train_' + ds + '.csv', index=False)
    y_test.to_csv('splittedDatasets/y_test_' + ds + '.csv', index=False)

    return x_train, x_test, y_train, y_test


datasets = ['income', 'titanic', 'social'] # names of datasets
modes = ['gaussian_copula', 'ctGAN', 'copulaGAN'] # available synthetic data methods

datasets = ['income']

def main():
    for ds in datasets:
        x_train, x_test, y_train, y_test = splitOriginalData(ds)
        trainEvaluateData(x_train, x_test, y_train, y_test, ds = ds, what='', classifier='forest', norm_confusion = 'true')
        for mode in modes:
            x_train_s, y_train_s = generateSyntheticData(ds, mode=mode, num_sample=5000)
            trainEvaluateData(x_train_s, x_test, y_train_s, y_test, ds = ds, what='Syntethic_' + mode )















def plot_line_results(results, dataset):

    if not os.path.isdir('Plots/comparison/'):
        os.mkdir('Plots/comparison/')

    fs = 15

    to_plot = {}

    for cl in ['adult']:
        to_plot[cl] = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        for i in ['recall','f1','precision']:
            print(dataset,' ',cl, ' ', i )
            to_plot[cl][i].append(results[dataset][cl][i])

    for i in ['recall','f1','precision']:

        fig, ax = plt.subplots(figsize = (12,5))

        plt.scatter(range(1,len(to_plot['adult'][i])+1), to_plot['adult'][i] )
        plt.plot(range(1,len(to_plot['adult'][i])+1), to_plot['adult'][i], label = 'adult' )


        plt.ylim(0,0.4)
        ax.set_xticks(range(1,len(to_plot['adult'][i])+1))
        ax.set_xticklabels(rotation=35, fontsize=10 )

        plt.title( i + ' for ' + dataset + ' dataset ', fontsize = fs )
        plt.legend()
        plt.tight_layout()
        plt.grid(ls=':', color='lightgray')
        plt.savefig('Plots/comparison/'+ i + '_comparison_'+ dataset + '.png', dpi = 200)
        plt.close()

if __name__=="__main__":
    main()



