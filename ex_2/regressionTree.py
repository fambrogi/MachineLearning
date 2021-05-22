from sklearn.model_selection import cross_val_score

import utilities as util
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from utilities import regressionErrors

#A node is initialized given its version of the dataset the attribute and value associated with that node
#computed by the father node
#given the values that it computes cof and number of rows it will decide if split again and generate new nodes or became
#a leaf

class Node:
    def __init__(self,dataset,attribute,value,target):
        self.dataset=dataset
        self.attribute=attribute
        self.value=value
        self.target=target
        self.avg=util.getAvg(dataset,target)
        self.standardDeviation=util.standardDeviation(dataset,target)
        self.childList=[]
        self.cof=self.standardDeviation/self.avg
        self.numberOfRows=dataset.shape[0]
        self.split()

    def split(self):
        if (self.numberOfRows<8 or self.cof< 0.1):
            return
        else:
            splitAttribute=util.getSplitAttribute(self.dataset,self.target)
            valueAverage=sum(self.dataset[splitAttribute])/self.numberOfRows
            left=Node(self.dataset.loc[self.dataset[splitAttribute]<valueAverage],splitAttribute,valueAverage,self.target)
            right=Node(self.dataset.loc[self.dataset[splitAttribute]>valueAverage],splitAttribute,valueAverage,self.target)
            '''
            dictionary=util.getAttributesValues(self.dataset)
            values=dictionary.get(splitAttribute)
            for value in values:
                self.childList.append(Node(util.getValues(self.dataset,splitAttribute,value),splitAttribute,value,self.target))
            '''
            self.childList.append(left)
            self.childList.append(right)

    def print(self):
        print(self.attribute+" "+str(self.value))
        if len(self.childList) != 0:
            for child in self.childList:
                child.print()
        else:
            print("END")


class Root:
    def __init__(self,dataset,target):
        self.dataset=dataset
        self.target=target
        self.avg=util.getAvg(dataset,target)
        self.standardDeviation=util.standardDeviation(dataset,target)
        self.childList=[]
        self.cof=self.standardDeviation/self.avg
        self.numberOfRows=dataset.shape[0]
        self.split()

    def split(self):
        if (self.numberOfRows < 8 or self.cof < 0.1):
            return
        else:
            splitAttribute = util.getSplitAttribute(self.dataset, self.target)
            valueAverage = sum(self.dataset[splitAttribute]) / self.numberOfRows
            left = Node(self.dataset.loc[self.dataset[splitAttribute] < valueAverage], splitAttribute, valueAverage, self.target)
            right = Node(self.dataset.loc[self.dataset[splitAttribute] >= valueAverage], splitAttribute, valueAverage, self.target)

            '''
            dictionary=util.getAttributesValues(self.dataset)
            values=dictionary.get(splitAttribute)
            for value in values:
                self.childList.append(Node(util.getValues(self.dataset,splitAttribute,value),splitAttribute,value,self.target))
            '''
            self.childList.append(left)
            self.childList.append(right)

    def print(self):
        for child in self.childList:
            child.print()




"""
def main():
    dataset = pd.read_csv("data/" + 'student-mat.csv')
    target = 'G1'
    print('creating tree')
    root=Root(dataset,target)
    print(root.print())
"""


def sk_regression(train_df, test_df, target):
    """ Trains the model using the DecisionTreeRegressor from sklearn """
    train_df_x = train_df.drop(columns = [target])
    train_df_y = train_df[target]
    test_df_x = test_df.drop(columns = [target])
    test_df_y = test_df[target]

    predictions = []
    criteria = ['mse','mae']

    for c in criteria: # different criteria for splitting
        regressor = DecisionTreeRegressor(random_state=0, criterion=c)
        regressor.fit(train_df_x, train_df_y)

        y_pred = regressor.predict(test_df_x)
        predictions.append(y_pred)
        #cross_val_score(regressor, test_df_x, test_df_y, cv=kfolds)

    # returns test_df_y, predictions, ['mse','mae','poisson']
    return test_df_y, predictions, criteria



