from sklearn.model_selection import train_test_split, RepeatedKFold
import utilities as util
from utilities import *

import pandas as pd
import numpy as np
import regressionTree as tree
from regressionTree import sk_regression
import modelTree
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
from clean_analyze_data import load_clean_data, data



#the method splits the dataset in train set and test set
"""
def split(dataset):
    train,test = train_test_split(dataset, test_size=0.3,shuffle=True)
    return train,test
"""

#the method implements the cross validation split
def crossSplit(dataset,folds):
    trainSets=[]
    testSets=[]
    kfold=RepeatedKFold(n_splits=folds, n_repeats=1, random_state=None)
    for train_index, test_index in kfold.split(dataset):
       trainSets.append(dataset.iloc[train_index, :])
       testSets.append(dataset.iloc[test_index, :])
    return trainSets,testSets


def prepareTest(dataset, target):
    targetCol = dataset[target]
    return targetCol, dataset

#given the train set the regression tree is created
def train(dataset,target):
    root=tree.Root(dataset,target)
    return root



#for every node in the child list of the node
#I search for the corrisponding value in the list
#if I find it I recall the method with that node
#If the child list is empty I assign to the row the avg value of the leaf node
#and set assigned to True
#we have to consider 3 cases
#the courrent node has the same value as the row so we have to check the child list and go down
#the current node hasn't the same value as the node we have to move to its sibling
#the current node is a leaf so we assign the value
def assignValue(row,node,assigned,target):

    #given a node I check if i'm in the right branch in this case I go deep
    for attribute in row.index:
        if attribute == node.attribute and row[attribute]==node.value:
            if (len(node.childList) == 0):
                return node.avg
            else:
                for child in node.childList:
                    assigned= assignValue(row, child, assigned,target)
                    if assigned != -1 and assigned != None:
                        break
        if assigned != -1 and assigned != None:
            return assigned



#for each row in the testset the method calls assignValue
def test(testSet,target,treeHead):
    assigned=-1
    results=[]
    for i in testSet.index:
        row=testSet.loc[i]
        for node in treeHead.childList:
            if(assigned == -1):
                assigned=assignValue(row,node,assigned,target)
                if(assigned == None):
                    assigned=node.avg
                if assigned != -1 and assigned != None:
                        results.append(assigned)
                        assigned=-1
                        break
            else:
                assigned=-1

    return results






def run(ds, folds):
    """ Wrapper function to train the model on the input dataset and target feature """
    # todo  here it goes the data cleaning !!!

    """ Reading, cleaning, splitting the data """
    dataset = load_clean_data(ds)

    for target in data[ds]['targets']:

        print('*** Training the dataset on the target: ', target , ' using ' , folds , ' folds cross-validation ')
        trainList, testList = crossSplit(dataset, folds)

        errors_tree = []
        errors_model = []

        y_test_sk_all, y_pred_sk_all = [], []
        y_pred_tree, y_pred_model = [], []

        for i in range(len(trainList)):

            print('*** Calculating Fold: ', i )
            print(' - training ')
            root = train(trainList[i], target)
            modelTreeRoot=modelTree.Root(trainList[i],target)

            #solCol, testSet = prepareTest(testList[i], target) # prepareTest func is useless?
            y_test, testSet = testList[i][target].values, testList[i]

            print(' - testing')
            y_pred = test(testSet, target, root)
            y_pred_ModelTree=test(testSet,target, modelTreeRoot)

            y_pred_tree.extend(y_pred)
            y_pred_model.extend(y_pred_ModelTree) # sono costanti ???

            """ Saving the errors for plotting """
            mse_rmse_mae_regressionTree = regressionErrors(y_pred, y_test)
            mse_rmse_mae_modelTree = regressionErrors(y_pred_ModelTree, y_test)

            errors_tree.append(mse_rmse_mae_regressionTree)
            errors_model.append(mse_rmse_mae_modelTree)

            print('*** Fold MSE, RMSE, MAE regression tree: ', mse_rmse_mae_regressionTree )
            print('*** Fold MSE, RMSE, MAE model tree: '     , mse_rmse_mae_modelTree)

            """ Using skregression """
            # returns test_df_y, predictions, ['mse','mae','poisson']
            y_test_sk, predictions_sk, criteria = sk_regression(trainList[i], testList[i], target)
            for p,c in zip(predictions_sk,criteria): # differnet criteria for splitting in sklearn
                mse_rmse_mae_sk = regressionErrors(y_test_sk, p)
                print('*** Fold MSE, RMSE, MAE sklearn for ', c , ' :', mse_rmse_mae_sk)
            y_test_sk_all.extend(y_test_sk)
            y_pred_sk_all.extend(predictions_sk[0])

        dummy_make_plot = plot_rms(errors_tree, errors_model, ds, target)
        dummy_diff = plot_diff(y_test_sk_all, y_pred_sk_all, y_pred_tree, criteria[0], ds, target)

        print('*** Done Fold: ', i)






""" # data as imported from clean_analyze_data

data = {'math': {'path': 'data/student-mat.csv',
                 'features': [],
                 'targets' : ['G1', 'G2', 'G3']},

        'life': { 'path': 'data/Life_Expectancy_Data.csv',
                  'features': ['AdultMortality',
                               'infantdeaths', 'Alcohol', 'percentageexpenditure', 'HepatitisB',
                               'Measles', 'BMI', 'under-fivedeaths', 'Polio', 'Totalexpenditure',
                               'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness1-19years',
                               'thinness5-9years', 'Incomecompositionofresources', 'Schooling' ],
                  'targets' : ['Lifeexpectancy']},
        }

"""


""" Folds for cross-validation """
folds = 2
datasets = ['wind']



if __name__ == '__main__':
    """ Selecting the datasets and respective targets """
    for ds in datasets: # these are the keys of the data dictionary i.e. names of the datasets
        run(ds, folds)







