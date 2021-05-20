import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
    reductionList=getReductions(dataframe, target)
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









def regressionErrors(testCol,solutionCol):
    """ Calculate MSE (mean squared error), RMSE (roor-MSE), MAE(mean absolute error) """


    differences = (solutionCol.values - testCol )
    MSE = np.mean(differences**2)
    RMSE = np.sqrt(np.mean(differences**2))
    MAE = np.mean(np.absolute(differences) )


    """ # manual formula for checking     
    sum=0
    for s,t in zip(solutionCol.values, testCol) :
        difference= s -t
        squaredDiff=difference**2
        sum+=squaredDiff
    mean2=sum/solutionCol.shape[0]

    # wrong due to pop removing last item in the array
    for i in solutionCol.index:
        regressorResult=testCol.pop()
        realResult=solutionCol.loc[i]
        #observed-predicted
        difference=realResult-regressorResult
        squaredDiff=difference**2
        sum+=squaredDiff
    mean=sum/solutionCol.shape[0]
    """

    return [MSE, RMSE, MAE]



"""
For Plotting
"""