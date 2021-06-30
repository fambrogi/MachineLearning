from sklearn.model_selection import train_test_split

import utilities as util
import pandas as pd
import numpy as np





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
        self.avg = util.getAvg(dataset, target)
        self.numberOfRows=dataset.shape[0]
        self.childList = {}

        # return if the target attribute has only a unique value
        if self.dataset[self.target].unique().shape[0] == 1:
            self.loss = util.loss(np.array(self.dataset[self.target]), np.repeat([self.avg], self.dataset[self.target].shape[0]))
            return

        self.standardDeviation=util.standardDeviation(dataset,target)
        self.cof=self.standardDeviation/self.avg
        #creating the model and computing the loss of the node
        self.model=util.getLinearClassifier()

        # check if train test split would actually result in usable data sets (if not set loss to minimal and return)
        if 0.3*dataset.shape[0] < 1:
            self.loss = util.loss(np.array(self.dataset[self.target]), np.repeat([self.avg], self.dataset[self.target].shape[0]))
            return
            #print("Endnode average: ", self.avg)
            #print("Endnode data point values: ", self.dataset[target])
        else:
            X_train, X_test, y_train, y_test = train_test_split(dataset, dataset[target], test_size=0.3,train_size=0.7,shuffle=True)
            util.fitLinearRegressor(X_train, y_train, self.model)
            self.prediction = util.predict(self.model, X_test)
            self.loss = util.loss(y_test, self.prediction)
            self.split()

    def split(self):
        if (self.numberOfRows<5 or self.cof< 0.05):
            self.loss = util.loss(np.array(self.dataset[self.target]), np.repeat([self.avg], self.dataset[self.target].shape[0]))
            return
        else:
            self.attribute = util.getSplitAttribute(self.dataset,self.target)
            self.value = np.average(self.dataset[self.attribute])
            # self.attribute = bestSplit["attribute"]
            # self.value = bestSplit["value"]
            if self.dataset.loc[self.dataset[self.attribute]<self.value].shape[0] == 0 or self.dataset.loc[self.dataset[self.attribute]>=self.value].shape[0] == 0:
                return
            left = Node(self.dataset.loc[self.dataset[self.attribute] < self.value], self.target)
            right = Node(self.dataset.loc[self.dataset[self.attribute] >= self.value], self.target)
            # build up the subtree error (depth first approach of the tree build makes left.loss and right.loss already contain the errors of deeper levels)
            lossSplit = (left.numberOfRows * left.loss + right.numberOfRows * right.loss) / self.numberOfRows
            self.childList["left"] = left
            self.childList["right"] = right
            if (lossSplit >= self.loss):
                self.childList = {}
            else:
                self.loss = lossSplit

    def print(self):
        print(self.attribute+" "+str(self.value))
        if(len(self.childList)!=0):
            for child in self.childList:
                self.childList[child].print()
        else:
            print("END")


# class Root:
#     def __init__(self,dataset,target):
#         self.dataset=dataset
#         self.target=target
#         self.avg = util.getAvg(dataset, target)
#         self.standardDeviation = util.standardDeviation(dataset, target)
#         self.childList = {}
#         self.cof = self.standardDeviation / self.avg
#         self.numberOfRows = dataset.shape[0]
#         # creating the model and computing the loss of the node
#         self.model = util.getLinearClassifier()
#         X_train, X_test, y_train, y_test = train_test_split(dataset,dataset[target], test_size=0.3, shuffle=True)
#         util.fitLinearRegressor(X_train, y_train, self.model)
#         self.prediction = util.predict(self.model, X_test)
#         self.loss = util.loss(y_test, self.prediction)
#         self.split()
#
#     def split(self):
#         if (self.numberOfRows < 20 or self.cof < 0.1):
#             return
#         else:
#            splitAttribute = util.getSplitAttribute(self.dataset, self.target)
#            self.attribute = splitAttribute
#            valueAverage = sum(self.dataset[splitAttribute]) / self.numberOfRows
#            self.value = valueAverage
#            left = Node(self.dataset.loc[self.dataset[splitAttribute] < valueAverage], self.target)
#            right = Node(self.dataset.loc[self.dataset[splitAttribute] >= valueAverage], self.target)
#             # build up the subtree error (depth first approach of the tree build makes left.loss and right.loss already contain the errors of deeper levels)
#            lossSplit = (left.numberOfRows * left.loss + right.numberOfRows*right.loss) / self.numberOfRows
#            self.childList["left"] = left
#            self.childList["right"] = right
#            '''
#            if (lossSplit >= self.loss):
#                self.childList = {}
#            else:
#                self.loss = lossSplit
#             '''

    def print(self):

        for child in self.childList:
            self.childList[child].print()
