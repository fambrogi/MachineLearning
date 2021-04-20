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

    #for col in notNumericCols:
    #    print(col)
    #    dataset[col].value_counts().plot.bar()
    #    plt.title(col)
    #    plt.plot()
    #    plt.savefig("preprocessingPlots/" + col + '.png', dpi=150)
    #    plt.close()

    # due the fact that numerical attributes are numerical but they corresponts at intervals I print histograms also for them in order to
    # spot outliers
    for col in numericCols:
        dataset[col].value_counts().plot.bar()
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
        if top_pct > 0.98:
            lowInfoCols.append(col)
            print('{0}: {1:.5f}%'.format(col, top_pct * 100))
            print(counts)
            print()
    return lowInfoCols

def findDuplicatedValues(dataset):
    df_dedupped = dataset.drop('RowID', axis=1).drop_duplicates()
    # there were duplicate rows
    print("finding duplicates")
    print(dataset.shape)
    print(df_dedupped.shape)

def main():
    testSet = pd.read_csv("data/advertisingBidding.shuf.tes.csv")
    trainSet = pd.read_csv("data/advertisingBidding.shuf.lrn.csv")
    solutionSet = pd.read_csv("data/advertisingBidding.shuf.sol.ex.csv")
    generalFrame=pd.concat([trainSet,testSet])
    printBasicInfo(generalFrame)

    findMissingValues(generalFrame)
    findDuplicatedValues(generalFrame)
    lowInfoCols=findLowInfoCols(generalFrame)



    #in the Url the 4% of the observation is null I delete those rows they are 1000 on 25000
    cleanedTrain=generalFrame[trainSet['URL'].notnull()]
    rowsToDelete=generalFrame[trainSet['URL'].isnull()].loc[:,'RowID']
    cleanedTest=generalFrame[testSet['URL'].notnull()]
    # I have to remove the rows in the solution set that I have deleted from the testSet
    cleanedSolution=solutionSet[~solutionSet.RowID.isin(rowsToDelete.tolist())]
    #The browser column creates issues due its values I want to solve it encoding the column
    labelencoder=LabelEncoder()

    cleanedTest['Browser_cat']=labelencoder.fit_transform(cleanedTest['Browser'])
    cleanedTrain['Browser_cat']=labelencoder.fit_transform(cleanedTrain['Browser'])
    cleanedTest['Adslotvisibility_cat'] = labelencoder.fit_transform(cleanedTest['Adslotvisibility'])
    cleanedTrain['Adslotvisibility_cat'] = labelencoder.fit_transform(cleanedTrain['Adslotvisibility'])

    cleanedTest['Adslotformat_cat'] = labelencoder.fit_transform(cleanedTest['Adslotformat'])
    cleanedTrain['Adslotformat_cat'] = labelencoder.fit_transform(cleanedTrain['Adslotformat'])

    del cleanedTest['Browser']
    del cleanedTrain['Browser']
    del cleanedTest['Adslotvisibility']
    del cleanedTrain['Adslotvisibility']
    del cleanedTest['Adslotformat']
    del cleanedTrain['Adslotformat']



    #I remove all this columns because we can assume that they depend by the single observation thay only create noise
    #and not relevant data
    del cleanedTest['RowID']
    del cleanedTest['UserID']
    del cleanedTrain['RowID']
    del cleanedTrain['UserID']
    del cleanedTest['BidID']
    del cleanedTrain['BidID']
    del cleanedTest['IP']
    del cleanedTrain['IP']

    del cleanedTest['Domain']
    del cleanedTrain['Domain']
    del cleanedTrain['URL']
    del cleanedTest['URL']
    del cleanedTrain['Time_Bid']
    del cleanedTest['Time_Bid']
    del cleanedTrain['AdslotID']
    del cleanedTest['AdslotID']
    del cleanedSolution['RowID']

    generalFrame = pd.concat([cleanedTrain, cleanedTest])
    numericC, nNumericC = devideNumericCols(generalFrame)
    #plotOutliers(generalFrame, nNumericC,numericC)
    cleanedTrain.to_csv('data/cleanedTrain.csv')
    cleanedTest.to_csv('data/cleanedTest.csv')
    cleanedSolution.to_csv('data/cleanedSolution.csv')

    print(cleanedTrain.head())

if __name__=="__main__":
    main()