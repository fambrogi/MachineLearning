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
            dictionary=util.getAttributesValues(self.dataset)
            values=dictionary.get(splitAttribute)
            for value in values:
                self.childList.append(Node(util.getValues(self.dataset,splitAttribute,value),splitAttribute,value,self.target))

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
            dictionary = util.getAttributesValues(self.dataset)
            values = dictionary.get(splitAttribute)
            for value in values:
                self.childList.append(Node(util.getValues(self.dataset,splitAttribute,value), splitAttribute, value, self.target))

    def print(self):

        for child in self.childList:
            child.print()
        return temp


def main():
    dataset = pd.read_csv("data/" + 'student-mat.csv')
    target = 'G1'
    print('creating tree')
    root=Root(dataset,target)
    print(root.print())


if __name__ == '__main__':
    main()


