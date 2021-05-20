from sklearn.model_selection import train_test_split, RepeatedKFold
import utilities as util
from utilities import *

import pandas as pd
import numpy as np
import regressionTree as tree
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
from clean_analyze_data import load_clean_data, data


#the method splits the dataset in train set and test set
def split(dataset):
    train,test=train_test_split(dataset, test_size=0.3,shuffle=True)
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


#given the train set the regression tree is created
def train(dataset,target):
    root=tree.Root(dataset,target)
    return root

def prepareTest(dataset,target):
    targetCol=dataset[target]
    return targetCol,dataset

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



def plot_rms(errors, ds_name, target):
    """ Plot punctual and averaged errors for each fold """

    os.system('mkdir Plots/results/')
    fs = 15
    for l,i,c in zip(['MSE', 'RMSE', 'MAE'], [0,1,2], ['lime', 'gold', 'blue']):

        plt.scatter(range(1,len(errors)+1), [f[i] for f in errors], label=l, color = c )
        plt.plot(range(1,len(errors)+1), np.full(len(errors), np.mean([g[i] for g in errors])),
                 label='Average', ls='--', color = c )

    plt.xlabel('K-fold')
    plt.legend(fontsize=7)
    plt.grid(ls=':', color='lightgray')
    plt.title('Dataset ' + ds_name + ' - Target feature: ' + target, fontsize=fs)

    plt.xticks(np.arange(1, len(errors)+1, 1.0))

    plt.savefig('Plots/results/' + ds_name + '_' + target + '.png', dpi=150 )
    plt.close()


def run(ds, folds):
    """ Wrapper function to train the model on the input dataset and target feature """
    # todo  here it goes the data cleaning !!!

    """ Reading, cleaning, splitting the data """
    dataset = load_clean_data(ds)

    trainSet, testSet = split(dataset)

    for target in data[ds]['targets']:

        print('*** Training the dataset on the target: ', target , ' using ' , folds , ' folds cross-validation ')
        trainList, testList = crossSplit(dataset, folds)

        errors = []

        for i in range(len(trainList)):

            print('*** Calculating Fold: ', i )
            print('training')
            root = train(trainList[i], target)
            solCol,testSet = prepareTest(testList[i], target)
            print('testing')
            results = test(testSet, target, root)
            print(results)
            """ Saving the errors for plotting """

            mse_rmse_mae = regressionErrors(results,solCol)
            errors.append(mse_rmse_mae)


            print('*** Fold MSE, RMSE, MAE: ', mse_rmse_mae )

        dummy_make_plot = plot_rms(errors, ds, target)

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
folds = 5
datasets = ['wind']


if __name__ == '__main__':

    """ Selecting the datasets and respective targets """
    for ds in datasets: # these are the keys of the data dictionary i.e. names of the datasets
        run(ds, folds)







