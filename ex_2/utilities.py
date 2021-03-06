import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
def getReductions2(dataframe,target):
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


'''
# calculates the value and SSR for the best split and returns them in a dictionary
def getBestSplit(df, column, target):
    bestValue = 0
    bestSSR = 999999999999

    # loop through all values of the column and calculate SSR. Only keep the best.
    for v in df[column].values:
        # split data
        lower = df.loc[df[column]<v]
        upper = df.loc[df[column]>=v]

        # if one of the dataframes is empty, continue
        if lower.shape[0] == 0 or upper.shape[0] == 0:
            continue

        # calculate the respective means
        lower_mean = np.mean(lower[target].values)
        upper_mean = np.mean(upper[target].values)

        # calculate residues
        lower_residues = lower[target].values - lower_mean
        upper_residues = upper[target].values - upper_mean

        # calculate SSR
        vSSR = np.sum(lower_residues**2) + np.sum(upper_residues**2)

        # check if this is a new best
        if vSSR < bestSSR:
            bestSSR = vSSR
            bestValue = v

    return {"attribute":column, "value":bestValue, "SSR":bestSSR}
'''

def getBestSplit(df, column, target, fast = False):

    # print("Length df: ", len(df) )

    def plot_ssr(values, rss, column):
        fs = 13
        if not os.path.isdir('Plots/results/rss'):
            os.mkdir('Plots/results/rss')

        plt.plot(values, rss)
        plt.xlabel(column, fontsize = fs)
        plt.ylabel('SSR', fontsize = fs)

        plt.title('SSR for the feature ' + column, fontsize = fs)
        plt.grid(ls=':', color='lightgray')
        column = column.replace('/', '_')
        plt.savefig('Plots/results/rss/' + column + '_rss.png', dpi=150)
        plt.close()


    bestValue = 0
    bestSSR = 999999999999

    # loop through all values of the column and calculate SSR. Only keep the best.

    # sort the feature values and the target in the same order
    # so I can loop over the feature values and do not need to search for indices
    values = df[column].values
    feature_indices = np.argsort(values)
    target_values = df[target].values
    sorted_target = target_values[feature_indices]
    indices_all = range(len(sorted_target))
    all_v, all_rss = [],[]

    if fast:
        values = df[column].values
        #values, indices = np.unique(df[column].values, return_index=True)  # extracting unique values
        sorted_indices = np.argsort(values) # sorting values
        values, indices =  np.unique(values[sorted_indices], return_index=True) # sorting unique indices in same order
        sorted_target = target_values[sorted_indices]
        indices_all = indices

    for ind,v in zip(indices_all, values):
        # split data
        #print('*** evaluating rss feature: ', column)
        lower = sorted_target[:ind]
        upper = sorted_target[ind:]

        # if one of the dataframes is empty, continue
        if len(lower) == 0 or len(upper) == 0:
            continue

        """
        # calculate the respective means
        lower_mean = np.mean(lower)
        upper_mean = np.mean(upper)

        # calculate residues
        lower_residues = lower - lower_mean
        upper_residues = upper - upper_mean

        # calculate SSR
        vSSR = np.sum(lower_residues**2) + np.sum(upper_residues**2)
        """

        rss_upper = np.sum(np.square(upper - np.mean(upper)))
        rss_lower = np.sum(np.square(lower - np.mean(lower)))
        vSSR = rss_lower + rss_upper

        all_rss.append(vSSR)
        all_v.append(v)

        # check if this is a new best
        if vSSR < bestSSR:
            bestSSR = vSSR
            bestValue = v

    dummy_plot = plot_ssr(all_v, all_rss, column)

    return {"attribute":column, "value":bestValue, "SSR":bestSSR}


# the method returns the attribute-value on which we have to split the dataset
def getSplitAttribute2(df,target):
    bestSplit = {"attribute":"", "value":0, "SSR":999999999999}

    # loop through all attributes and get the best split for each. Only keep the best.
    for col in df.columns:
        if col == target or col in ['G1','G2','G3']: # must remove all targets for math
            continue
        curr_bestSplit = getBestSplit(df, col, target)
        if curr_bestSplit["SSR"] < bestSplit["SSR"]:
            bestSplit = curr_bestSplit

    return bestSplit

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
    if(indexMax<targetIndex):
        toRet = dataframe.columns[indexMax]
        return toRet
    else:
        toRet = dataframe.columns[indexMax+1]
        return toRet


def getLinearClassifier():
    return LinearRegression()

def getRandomForestRegressor():
    return RandomForestRegressor()

def fitLinearRegressor(x,y,regressor):
    regressor.fit(x,y)

def predict(regressor, X):
    return regressor.predict(X)

def loss(y, y_pred):
        return mean_squared_error(y, y_pred)


def regressionErrors(y_test, y_pred):
    """ Calculate MSE (mean squared error), RMSE (roor-MSE), MAE(mean absolute error) """

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    return [MSE, RMSE, MAE]



"""
For Plotting
"""


def plot_rms(errors_tree, errors_model,
             mse_rmse_mae_sk, mse_rmse_mae_linear, mse_rmse_mae_rf,
            ds, target):
    """ Plot punctual and averaged errors for each fold """

    os.system('mkdir Plots/results/')
    fs = 12
    # ['lime', 'blue', 'gold', 'red', 'black']
    for l,i in zip(['MSE', 'RMSE', 'AME'], [0,1,2]):

        plt.scatter(range(1,len(errors_tree)+1), [f[i] for f in errors_tree], label= '*Reg. tree', color = 'lime' )
        plt.plot(range(1,len(errors_tree)+1), np.full(len(errors_tree), np.mean([g[i] for g in errors_tree])), ls=':', color = 'lime' )

        plt.scatter(range(1,len(errors_model)+1), [f[i] for f in errors_model], label='*Model tree', color = 'blue' )
        plt.plot(range(1,len(errors_model)+1), np.full(len(errors_model), np.mean([g[i] for g in errors_model])), ls=':', color = 'blue' )

        plt.scatter(range(1,len(mse_rmse_mae_sk)+1), [f[i] for f in mse_rmse_mae_sk], label='Sci-kit Reg. Tree', color = 'gold' )
        plt.plot(range(1,len(mse_rmse_mae_sk)+1), np.full(len(mse_rmse_mae_sk), np.mean([g[i] for g in mse_rmse_mae_sk])), ls=':', color = 'gold' )

        plt.scatter(range(1,len(mse_rmse_mae_linear)+1), [f[i] for f in mse_rmse_mae_linear], label= 'Sci-kit Lin. Reg.', color = 'red' )
        plt.plot(range(1,len(mse_rmse_mae_linear)+1), np.full(len(mse_rmse_mae_linear), np.mean([g[i] for g in mse_rmse_mae_linear])), ls=':', color = 'red')

        plt.scatter(range(1,len(mse_rmse_mae_rf)+1), [f[i] for f in mse_rmse_mae_rf], label=l + 'Sci-kit Rand. Forest', color = 'black' )
        plt.plot(range(1,len(mse_rmse_mae_rf)+1), np.full(len(mse_rmse_mae_rf), np.mean([g[i] for g in mse_rmse_mae_rf])), ls=':', color = 'black' )

        plt.xlabel('K-fold')
        plt.legend(fontsize=7)
        plt.grid(ls=':', color='lightgray')
        plt.title('Dataset ' + ds + ' - Target feature: ' + target, fontsize=fs)

        plt.xticks(np.arange(1, len(errors_tree)+1, 1.0))
        #plt.tight_layout()
        plt.ylabel(l, fontsize = fs)
        plt.savefig('Plots/results/' + ds + '_' + target + '_' + l + '.png', dpi=150 )
        plt.close()


def plot_diff(y_test_sk, y_pred_tree, y_pred_ModelTree,
              y_pred_skTree, y_pred_randomF, prediction_linReg, criterion, ds, target):

    num_points = 100
    fs = 12
    plt.plot(y_test_sk[:num_points], label = 'Test values', ls = ':', color = 'black')
    plt.scatter(range(num_points), y_test_sk[:num_points], ls = ':', color = 'black')

    plt.scatter(range(num_points), y_pred_skTree[:num_points],
                label = 'sci-kit Regr. Tree '  , color = 'red')

    plt.scatter(range(num_points), y_pred_tree[:num_points],
                label = '*Regr. Tree' , color = 'blue')

    plt.scatter(range(num_points), y_pred_randomF[:num_points],
                label = '* sci-kit Random Forest' , color = 'gold')

    plt.title('Test set vs Predictions - ' + ds, fontsize=fs)
    plt.ylabel(target, fontsize=fs)
    plt.xlabel('Test item', fontsize = fs )
    plt.legend(fontsize=fs-3, loc = 'best')
    #plt.tight_layout()
    plt.grid(ls=':', color = 'lightgray')


    plt.savefig('Plots/results/' + ds + '_sklearn_comparison_lines.png', dpi = 150)
    plt.close()


    plt.hist([y_test_sk, y_pred_tree, y_pred_ModelTree,  y_pred_skTree, y_pred_randomF, prediction_linReg ],
             histtype = 'step',

             label = ['Test values', '*Regr. Tree', '*Model Tree',
                      'sci-kit Regr. ', 'sci-kit Random Forest' , 'sci-kit Lin. Regr.' ],

             color = ['blue','lime','orange', 'red', 'black', 'cyan'],
             density = True )

    plt.grid(ls=':', color='lightgray')
    plt.legend(fontsize=fs-4, loc = 'best' )
    plt.ylabel("Normalized Counts" , fontsize=fs)

    #plt.tight_layout()
    if ds == 'wind' or ds == 'math':
        plt.xlim(0,20)
        plt.xlabel(target, fontsize=fs)
    else:
        plt.xlabel(target, fontsize=fs)

    plt.savefig('Plots/results/' + ds + '_sklearn_comparison_histo_' + target + '.png', dpi=150)
    plt.close()
