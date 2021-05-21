import pandas as pd
import numpy as np


dataset = pd.read_csv("data/" + 'student-mat.csv')
target=['G1','G2','G3']

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

#the method computes for each attribute the standard deviation reduction and puts them in a list
def getReductions(dataframe,target):
    valuesDictionary = getAttributesValues(dataframe)
    reductionsList=[]
    for attribute in valuesDictionary.keys():
        values = valuesDictionary.get(attribute)
        reductionsList.append(getStandardDeviationReduction(dataset, attribute, values, target))
    return reductionsList

# the method returns the attribute on which we have to split the dataset
def getSplitAttribute(dataframe,target):
    reductionList=getReductions(dataset, 'G1')
    maxReduction=max(reductionList)
    indexMax=reductionList.index(maxReduction)
    return dataframe.columns[indexMax]

