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
    def __init__(self,dataset,target):
        self.dataset=dataset
        self.target=target
        self.attribute=""
        self.value=0
        self.avg=util.getAvg(dataset,target)
        self.numberOfRows=dataset.shape[0]
        self.childList={}

        # return if the target attribute has only a unique value
        if self.dataset[self.target].unique().shape[0] == 1:
            self.loss = util.loss(np.array(self.dataset[self.target]), np.repeat([self.avg], self.dataset[self.target].shape[0]))
            return

        self.standardDeviation=util.standardDeviation(dataset,target)
        self.cof=self.standardDeviation/self.avg
        self.split()

    def split(self):
        if (self.numberOfRows<5 or self.cof< 0.05):
            return
        else:
            self.attribute = util.getSplitAttribute(self.dataset,self.target)
            self.value = np.average(self.dataset[self.attribute])
            # self.attribute = bestSplit["attribute"]
            # self.value = bestSplit["value"]
            # check if split would be redundant
            if self.dataset.loc[self.dataset[self.attribute]<self.value].shape[0] == 0 \
                    or self.dataset.loc[self.dataset[self.attribute]>=self.value].shape[0] == 0:
                return
            left = Node(self.dataset.loc[self.dataset[self.attribute]<self.value],self.target)
            right = Node(self.dataset.loc[self.dataset[self.attribute]>=self.value],self.target)
            '''
            dictionary=util.getAttributesValues(self.dataset)
            values=dictionary.get(self.attribute)
            for value in values:
                self.childList.append(Node(util.getValues(self.dataset,self.attribute,value),self.attribute,value,self.target))
            '''
            self.childList["left"] = left
            self.childList["right"] = right

    def print(self):
        print(self.attribute+" "+str(self.value))
        if len(self.childList) != 0:
            for child in self.childList:
                self.childList[child].print()
        else:
            print("END")


# class Root:
#     def __init__(self,dataset,target):
#         self.dataset=dataset
#         self.target=target
#         self.avg=util.getAvg(dataset,target)
#         self.standardDeviation=util.standardDeviation(dataset,target)
#         self.childList={}
#         self.cof=self.standardDeviation/self.avg
#         self.numberOfRows=dataset.shape[0]
#         self.split()
#
#     def split(self):
#         if (self.numberOfRows < 20 or self.cof < 0.1):
#             return
#         else:
#             splitAttribute = util.getSplitAttribute(self.dataset, self.target)
#             valueAverage = sum(self.dataset[splitAttribute]) / self.numberOfRows
#             left = Node(self.dataset.loc[self.dataset[splitAttribute] < valueAverage], self.target)
#             right = Node(self.dataset.loc[self.dataset[splitAttribute] >= valueAverage], self.target)
#
#             '''
#             dictionary=util.getAttributesValues(self.dataset)
#             values=dictionary.get(splitAttribute)
#             for value in values:
#                 self.childList.append(Node(util.getValues(self.dataset,splitAttribute,value),splitAttribute,value,self.target))
#             '''
#             self.childList["left"] = left
#             self.childList["right"] = right
#
#     def print(self):
#         for child in self.childList:
#             self.childList[child].print()




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
