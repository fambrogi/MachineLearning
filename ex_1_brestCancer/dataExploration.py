import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def printBasicInfo(dataset):
    # shape and data types of the data
    print(dataset.shape)
    print(dataset.dtypes)

def devideNumericCols(dataset):
    # select numeric columns
    datasetNumeric = dataset.select_dtypes(include=[np.number])
    numericCols = datasetNumeric.columns.values
    print(numericCols)
    # select non numeric columns
    datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
    notNumericColums = datasetNotNumeric.columns.values
    print(notNumericColums)
    return numericCols,notNumericColums

def findMissingValues(dataset):
    # missing values percentage
    for col in dataset.columns:
        pct_missing = np.mean(dataset[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing * 100)))

def plotOutliers(dataset,notNumericCols,numericCols):
    # outliers for categorical attributes

    for col in notNumericCols:
        print(col)
        dataset[col].value_counts().plot.bar()
        plt.title(col)
        plt.plot()
        plt.savefig("preprocessingPlots/" + col + '.png', dpi=150)
        plt.close()

    # due the fact that numerical attributes are numerical but they corresponts at intervals I print histograms also for them in order to
    # spot outliers
    for col in numericCols:
        dataset.boxplot(column=col)
        plt.title(col)
        plt.plot()
        plt.savefig("preProcessingplots/" + col + '.png', dpi=150)
        plt.close()

def findLowInfoCols(dataset):
    print('low information columns')
    numRows = len(dataset.index)
    lowInfoCols = []
    for col in dataset.columns:
        counts = dataset[col].value_counts(dropna=False)
        top_pct = (counts / numRows).iloc[0]
        if top_pct > 0.80:
            lowInfoCols.append(col)
            print('{0}: {1:.5f}%'.format(col, top_pct * 100))
            print(counts)
            print()
    return lowInfoCols

def findDuplicatedValues(dataset):
    df_dedupped = dataset.drop('ID', axis=1).drop_duplicates()
    # there were duplicate rows
    print("finding duplicates")
    print(dataset.shape)
    print(df_dedupped.shape)

def main():
    testSet = pd.read_csv("Data/breast-cancer-diagnostic.shuf.tes.csv")
    trainSet = pd.read_csv("Data/breast-cancer-diagnostic.shuf.lrn.csv")
    solutionSet = pd.read_csv("Data/breast-cancer-diagnostic.shuf.sol.ex.csv")

    generalFrame=pd.concat([trainSet,testSet])

    printBasicInfo(generalFrame)

    findMissingValues(generalFrame)
    findDuplicatedValues(generalFrame)
    lowInfoCols=findLowInfoCols(generalFrame)

    print(lowInfoCols)

    #In this dataset the numerical values are not organized in ranges but we can have values in all the domain
    #we don't have intervals instead of instograms is better use boxplots
    generalFrame = pd.concat([trainSet, testSet])
    numericC, nNumericC = devideNumericCols(generalFrame)
    plotOutliers(generalFrame, nNumericC,numericC)

    cleanedTrain=trainSet
    cleanedTest=testSet
    cleanedSolution=solutionSet
    del cleanedTrain['ID']
    del cleanedTest['ID']
    del cleanedSolution['ID']
    cleanedTrain.to_csv('data/cleanedTrain.csv')
    cleanedTest.to_csv('data/cleanedTest.csv')
    cleanedSolution.to_csv('data/cleanedSolution.csv')

    print(cleanedTrain.head())


if __name__=="__main__":
    main()