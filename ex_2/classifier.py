from sklearn.model_selection import train_test_split, RepeatedKFold
import utilities as util
import pandas as pd
import numpy as np
import regressionTree as tree
from sklearn.model_selection import KFold


#the method splits the dataset in train set and test set
def split(dataset):
    train,test=train_test_split(dataset,test_size=0.3,shuffle=True)
    return train,test

#the method implements the cross validation split
def crossSplit(dataset,folds):
    trainSets=[]
    testSets=[]
    kfold=RepeatedKFold(n_splits=folds,n_repeats=1, random_state=None)
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

def rootMeanSquaredError(testCol,solutionCol):
    sum=0
    for i in solutionCol.index:
        regressorResult=testCol.pop()
        realResult=solutionCol.loc[i]
        #observed-predicted
        difference=realResult-regressorResult
        squaredDiff=difference**2
        sum+=squaredDiff
    mean=sum/solutionCol.shape[0]
    return np.sqrt(mean)


def main():
    dataset = pd.read_csv("data/" + 'student-mat.csv')
    target = 'G1'
    folds=5
    trainList,testList=crossSplit(dataset,5)
    rmsqList=[]
    for i in range(len(trainList)):
        print('training fold '+str(i))
        root = train(trainList[i], target)
        solCol,testSet=prepareTest(testList[i],target)
        print('testing fold ' + str(i))
        results=test(testSet,target,root)
        rmsq=rootMeanSquaredError(results,solCol)
        rmsqList.append(rmsq)
    print(rmsqList)
    rmsqAvg=sum(rmsqList)/folds
    print(rmsqAvg)

if __name__ == '__main__':
    main()








