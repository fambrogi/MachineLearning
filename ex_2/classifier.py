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

def split(dataset):
    train,test = train_test_split(dataset, test_size=0.3,shuffle=True)
    return train,test


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

    # if it is the end node, return its avgerage
    if (len(node.childList) == 0):
        return node.avg
    #given a node I check if i'm in the right branch in this case I go deep
    for attribute in row.index:
        if attribute == node.attribute:
            if row[attribute] < node.value:
                assigned = assignValue(row, node.childList["left"], assigned, target)
            else:
                assigned = assignValue(row, node.childList["right"], assigned, target)
        if assigned != -1:
            return assigned



#for each row in the testset the method calls assignValue
def test(testSet,target,treeHead):
    results=[]
    for i in testSet.index:
        assigned=-1
        row=testSet.loc[i]
        # print("length: ", len(treeHead.childList))
        assigned=assignValue(row,treeHead,assigned,target)
        results.append(assigned)
    return results




""" data as imported from clean_analyze_data
data = {'math': {'path': 'data/student-mat.csv',
                 'features': ['age', 'Medu', 'Fedu'],
                 'drop': [],
                 'targets' : ['G1', 'G2', 'G3']},

        'life': { 'path': 'data/Life_Expectancy_Data.csv',
                  'features': ['AdultMortality',
                               'infantdeaths', 'Alcohol', 'percentageexpenditure', 'HepatitisB',
                               'Measles', 'BMI', 'under-fivedeaths', 'Polio', 'Totalexpenditure',
                               'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness1-19years',
                               'thinness5-9years', 'Incomecompositionofresources', 'Schooling' ],
                  'drop': ['Country', 'Year', 'Status'],
                  'targets': ['Lifeexpectancy']},

        'wind': {'path': 'data/wind_train_data.csv',
                  'features': ['wind_speed(m/s)',
                               'atmospheric_temperature(??C)', 'shaft_temperature(??C)',
                               'blades_angle(??)', 'gearbox_temperature(??C)', 'engine_temperature(??C)',
                               'motor_torque(N-m)', 'generator_temperature(??C)',
                               'atmospheric_pressure(Pascal)', 'area_temperature(??C)',
                               'windmill_body_temperature(??C)', 'wind_direction(??)', 'resistance(ohm)',
                               'rotor_torque(N-m)', 'blade_length(m)',
                               'blade_breadth(m)', 'windmill_height(m)' ],

                  'drop': ['turbine_status', 'cloud_level', 'tracking_id', 'datetime'],
                  'targets': ['windmill_generated_power(kW_h)']},

        }
"""


""" Folds for cross-validation """
folds = 5
datasets = ['math']


if __name__ == '__main__':
    """ Selecting the datasets and respective targets """
    for ds in datasets:

        dataset = load_clean_data(ds)

        for target in data[ds]['targets']:

            print('*** Training the dataset on the target: ', target , ' using ' , folds , ' folds cross-validation ')
            trainList, testList = crossSplit(dataset, folds)

            errors_tree = []
            errors_model = []

            y_test_sk_all, y_pred_sk_all = [], []
            y_test_sk_all, y_pred_sk_all = [], []

            y_pred_tree, y_pred_model = [], []
            prediction_randomForest_all = []
            prediction_linReg_all = []

            for i in range(len(trainList)):

                print('*** Calculating Fold: ', i )
                print(' - training Regression Tree ')
                root = tree.Node(trainList[i],target)
                print(' - training Model Tree ')
                modelTreeRoot = modelTree.Node(trainList[i],target)

                solCol, testSet = prepareTest(testList[i], target)
                y_test, testSet = testList[i][target].values, testList[i]

                print(' - testing')
                y_pred = test(testSet, target, root)
                y_pred_ModelTree=test(testSet, target, modelTreeRoot)

                y_pred_tree.extend(y_pred)
                y_pred_model.extend(y_pred_ModelTree)
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

                """ Using linear regressor """
                X_train, X_test = trainList[i].loc[:, trainList[i].columns != target], testList[i].loc[:, testList[i].columns != target]
                y_train, y_test = trainList[i][target], testList[i][target]

                """ Federico version
                train_df = trainList[i]
                test_df = testList[i]

                X_train = train_df.drop(columns=[target])
                y_train = train_df[target]
                X_test = test_df.drop(columns=[target])
                y_test = test_df[target]
                """

                linear_regressor = util.getLinearClassifier()
                util.fitLinearRegressor(X_train, y_train, linear_regressor)
                prediction_linReg = util.predict(linear_regressor, X_test)
                mse_rmse_mae_linear = regressionErrors(y_test, prediction_linReg)
                print('*** Fold MSE, RMSE, MAE sklearn for linear regressor :', mse_rmse_mae_linear)
                prediction_linReg_all.extend(prediction_linReg)


                """ Using random forest regressor """
                # X_train, X_test = trainList[i].loc[:, trainList[i].columns != target], testList[i].loc[:, testList[i].columns != target]
                # y_train, y_test = trainList[i][target], testList[i][target]
                rf_regressor = util.getRandomForestRegressor()
                util.fitLinearRegressor(X_train, y_train, rf_regressor)
                prediction_randomForest = util.predict(rf_regressor, X_test)
                mse_rmse_mae_rf = regressionErrors(y_test, prediction_randomForest)
                print('*** Fold MSE, RMSE, MAE sklearn for random forest regressor :', mse_rmse_mae_rf)
                prediction_randomForest_all.extend(prediction_randomForest)

            dummy_make_plot = plot_rms(errors_tree, errors_model, ds, target)

            dummy_diff = plot_diff(y_test_sk_all,
                                   y_pred_tree,
                                   y_pred_ModelTree,
                                   y_pred_sk_all,
                                   prediction_randomForest_all,
                                   prediction_linReg_all,
                                   criteria[0], ds, target)

            print('************** Check difference in predictions: ' , '\n')
            print('Test set \t\t ', y_test_sk_all[:10]  , '\n')
            print('Regr. Tree \t\t ', y_pred_tree[:10]  , '\n')
            print('Model Tree \t\t ', y_pred_ModelTree[:10] , '\n')
            print('sci-kit Reg.Tree  \t\t ', y_pred_sk_all[:10] , '\n')
            print('sci-kit Random Forest  \t\t ', prediction_randomForest[:10] , '\n')
            print('sci-kit Lin. Regr.   \t\t ', prediction_linReg[:10] , '\n')

            print('*** Done Fold: ', i)
