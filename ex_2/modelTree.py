from sklearn.model_selection import train_test_split

import utilities as util
import pandas as pd
import numpy as np





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
        self.avg = util.getAvg(dataset, target)
        self.standardDeviation=util.standardDeviation(dataset,target)
        self.childList=[]
        self.cof=self.standardDeviation/self.avg
        self.numberOfRows=dataset.shape[0]
        #creating the model and computing the loss of the node
        self.model=util.getLinearClassifier()
        X_train, X_test, y_train, y_test = train_test_split(dataset, dataset[target], test_size=0.3, shuffle=True)
        util.fitLinearRegressor(X_train, y_train, self.model)
        self.prediction = util.predict(self.model, X_test)
        self.loss = util.loss(y_test, self.prediction)
        self.split()

    def split(self):
        if (self.numberOfRows<8 or self.cof< 0.1):
            return
        else:
            splitAttribute = util.getSplitAttribute(self.dataset, self.target)
            valueAverage = sum(self.dataset[splitAttribute]) / self.numberOfRows
            left = Node(self.dataset.loc[self.dataset[splitAttribute] < valueAverage], splitAttribute, valueAverage,
                        self.target)
            right = Node(self.dataset.loc[self.dataset[splitAttribute] >= valueAverage], splitAttribute,
                         valueAverage, self.target)
            lossSplit = (left.numberOfRows * left.loss) + (right.numberOfRows * right.loss)
            self.childList.append(right)
            self.childList.append(left)
            lossSplit = lossSplit / self.numberOfRows
            if (lossSplit >= self.loss):
                self.childList = []

    def print(self):
        print(self.attribute+" "+str(self.value))
        if(len(self.childList)!=0):
            for child in self.childList:
                child.print()
        else:
            print("END")


class Root:
    def __init__(self,dataset,target):
        self.dataset=dataset
        self.target=target
        self.target = target
        self.avg = util.getAvg(dataset, target)
        self.standardDeviation = util.standardDeviation(dataset, target)
        self.childList = []
        self.cof = self.standardDeviation / self.avg
        self.numberOfRows = dataset.shape[0]
        # creating the model and computing the loss of the node
        self.model = util.getLinearClassifier()
        X_train, X_test, y_train, y_test = train_test_split(dataset,dataset[target], test_size=0.3, shuffle=True)
        util.fitLinearRegressor(X_train, y_train, self.model)
        self.prediction = util.predict(self.model, X_test)
        self.loss = util.loss(y_test, self.prediction)
        self.split()

    def split(self):
        if (self.numberOfRows < 8 or self.cof < 0.1):
            return
        else:
           splitAttribute = util.getSplitAttribute(self.dataset, self.target)
           valueAverage = sum(self.dataset[splitAttribute]) / self.numberOfRows
           left = Node(self.dataset.loc[self.dataset[splitAttribute] < valueAverage], splitAttribute, valueAverage,
                            self.target)
           right = Node(self.dataset.loc[self.dataset[splitAttribute] >= valueAverage], splitAttribute,
                             valueAverage, self.target)
           lossSplit = (left.numberOfRows * left.loss )+(right.numberOfRows*right.loss)
           self.childList.append(right)
           self.childList.append(left)
           lossSplit = lossSplit / self.numberOfRows
           if (lossSplit >= self.loss):
               self.childList = []

    def print(self):

        for child in self.childList:
            child.print()







