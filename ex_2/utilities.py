import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from sklearn import metrics
import matplotlib.pyplot as plt


def standardDeviation(dataframe,target):

    return np.std(dataframe[target])

def getAvg(dataframe,target):
    return np.average(dataframe[target])

#the method returns a dictionary for
#each columns there is the list of all possible values
def getAttributesValues(dataframe):
    cols=dataframe.columns[0:-3]
    attributeList=[]
    dictionary={}
    for col in cols:
        dictionary[col]=list(set(dataframe[col].tolist()))
    return dictionary

#this method given an attribute and a values return all the rows associated with them
def getValues(dataframe,attribute,value):
    restrictedDataframe = dataframe[dataframe[attribute].isin([value])]
    del restrictedDataframe[attribute]
    return restrictedDataframe

#The following method will be called after getValues
#it returns the number of rows into the restrictedDataframe and the deviationStandard
def getDatasetInfo(dataframe,target):
    numRows=dataframe.shape[0]
    std=standardDeviation(dataframe,target)
    return numRows,std

'''
#given the attribute compute the weighted standard deviation
def getStandardDeviationReduction(dataframe,attribute,values,target):
    genNumberOfRows=getDatasetInfo(dataframe,target)[0]
    genStd=getDatasetInfo(dataframe,target)[1]
    stdList=[]
    numRows=[]
    for val in values:
        restrictedData=getValues(dataframe,attribute,val)
        stdList.append(getDatasetInfo(restrictedData,target)[1])
        numRows.append(getDatasetInfo(restrictedData,target)[0])
    weightedSum=0
    for i in range(len(stdList)):
        weightedSum+=(numRows[i]/genNumberOfRows)*stdList[i]
    return genStd-weightedSum
'''

def getStandardDeviationReduction(dataframe,attribute,avg,target):
    genNumberOfRows=getDatasetInfo(dataframe,target)[0]
    genStd=getDatasetInfo(dataframe,target)[1]
    stdList=[]
    numRows=[]
    left=dataframe.loc[dataframe[attribute] < avg]
    right=dataframe.loc[dataframe[attribute] >= avg]
    stdList.append(getDatasetInfo(left,target)[1])
    numRows.append(getDatasetInfo(left,target)[0])
    stdList.append(getDatasetInfo(right, target)[1])
    numRows.append(getDatasetInfo(right, target)[0])
    weightedSum=0
    for i in range(len(stdList)):
        weightedSum+=(numRows[i]/genNumberOfRows)*stdList[i]
    return genStd-weightedSum
#the method computes for each attribute the standard deviation reduction and puts them in a list
def getReductions(dataframe,target):
   '''
    valuesDictionary = getAttributesValues(dataframe)
    reductionsList=[]
    for attribute in valuesDictionary.keys():
        values = valuesDictionary.get(attribute)
        reductionsList.append(getStandardDeviationReduction(dataframe, attribute, values, target))
    return reductionsList
    '''

   reductionsList = []
   for attribute in dataframe.columns:
       valueAverage = np.average(dataframe[attribute])
       reductionsList.append(getStandardDeviationReduction(dataframe, attribute, valueAverage, target))
   return reductionsList


# the method returns the attribute on which we have to split the dataset
def getSplitAttribute(dataframe,target):
    targetIndex= dataframe.columns.get_loc(target)
    reductionList=getReductions(dataframe, target)
    del reductionList[targetIndex]
    maxReduction=max(reductionList)
    indexMax=reductionList.index(maxReduction)
    return dataframe.columns[indexMax]

def getLinearClassifier():
    return LinearRegression()

def fitLinearRegressor(x,y,regressor):
    regressor.fit(x,y)

def predict(regressor, X):
    return regressor.predict(X)

def loss(y, y_pred):
        return mean_squared_error(y, y_pred)


def regressionErrors(y_test, y_pred):
    """ Calculate MSE (mean squared error), RMSE (roor-MSE), MAE(mean absolute error) """

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    return [MSE, RMSE, MAE]



"""
For Plotting
"""


def plot_rms(errors_tree, errors_model, ds_name, target):
    """ Plot punctual and averaged errors for each fold """

    os.system('mkdir Plots/results/')
    fs = 12
    for l,i,c in zip(['MSE', 'RMSE', 'MAE'], [0,1,2], ['lime', 'gold', 'blue']):

        plt.scatter(range(1,len(errors_tree)+1), [f[i] for f in errors_tree], label=l + ' tree', color = c )
        plt.scatter(range(1,len(errors_model)+1), [f[i] for f in errors_model], label=l + ' model', color = c , ls = ':' )

        plt.plot(range(1,len(errors_tree)+1), np.full(len(errors_tree), np.mean([g[i] for g in errors_tree])),
                 label='Average Tree', ls=':', color = c )

    plt.xlabel('K-fold')
    plt.legend(fontsize=7)
    plt.grid(ls=':', color='lightgray')
    plt.title('Dataset ' + ds_name + ' - Target feature: ' + target, fontsize=fs)

    plt.xticks(np.arange(1, len(errors_tree)+1, 1.0))
    plt.tight_layout()
    plt.savefig('Plots/results/' + ds_name + '_' + target + '.png', dpi=150 )
    plt.close()

def plot_diff(y_test_sk, predictions_sk, y_pred_tree, criterion, ds, target):

    num_points = 300
    fs = 12
    plt.plot(y_test_sk[:num_points], label = 'Test values')
    plt.scatter(range(num_points), predictions_sk[:num_points], label = 'Predicted SKlearn ' + criterion)
    plt.scatter(range(num_points), y_pred_tree[:num_points], label = 'Predicted Tree' )

    plt.title('Test set vs Predictions - ' + ds, fontsize=fs)
    plt.ylabel(target, fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.grid(ls=':', color = 'lightgray')
    plt.savefig('Plots/results/sklearn_comparison_lines_' + ds + '.png', dpi = 150)
    plt.close()

    plt.hist([y_test_sk,predictions_sk,y_pred_tree],
             histtype = 'step',
             label = ['Test values', 'Predicted SKlearn ' + criterion , 'Predicted Tree' ],
             color = ['blue','lime','orange'])

    plt.grid(ls=':', color='lightgray')
    plt.ylabel(target, fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig('Plots/results/sklearn_comparison_histo_' + ds + '.png', dpi = 150)