from sklearn.model_selection import train_test_split
import utilities as util
import pandas as pd
import numpy as np
import regressionTree as tree


#the method splits the dataset in train set and test set
def split(dataset):
    train,test= train_test_split(dataset, test_size=0.3, shuffle=True )
    return train,test

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
                if assigned != -1:
                    if assigned== None:
                        results.append(0)
                    else:
                        results.append(assigned)
                    assigned=-1
                    break
            else:
                assigned=-1
                break
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



def run(dataset, targets):
    """ Wrapper function to train the model on the input dataset and target feature """
    # todo  here it goes the data cleaning !!!

    """ Reading, cleaning, splitting the data """
    print('*** Reading and preparing the dataset: ' , dataset )

    dataset = pd.read_csv(dataset)
    trainSet, testSet = split(dataset)

    for target in targets:
        print('*** Training the dataset: ', dataset, ' on the target: ', target)
        root = train(trainSet, target)

        print('*** Testing the dataset: ', dataset, ' on the target: ', target)
        solCol,testSet = prepareTest(testSet,target)
        results = test(testSet,target,root)
        rmsq = rootMeanSquaredError(results,solCol)
        print('*** Root mean square: ', rmsq )


""" Dictionary of the datasets """
data = {'data/student-mat.csv': ['G1', 'G2', 'G3'], }

""" Folds for cross-validation """
folds = 5
data = {'data/student-mat.csv': ['G1', 'G2', 'G3'], }

if __name__ == '__main__':
    """ Selecting the datasets and respective targets """
    for ds in data.keys():
        run(ds, data[ds])








