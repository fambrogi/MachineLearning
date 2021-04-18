import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier



def importDataset():
    dataset = pd.read_csv("data/cleanedDataset.csv")
    return dataset

def splitDataset(dataset,c):
    x = dataset[['age', 'gender', 'education', 'ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore',
                 'Cscore', 'Impulsive', 'SS']]
    y = dataset[c]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return x,y,x_train,x_test,y_train,y_test

def giniBasedClassifier(x_train,x_test,y_train):
    giniClassifier=DecisionTreeClassifier(criterion = "gini")
    giniClassifier.fit(x_train,y_train)
    return giniClassifier

def entropyBasedClassifier(x_train,x_test,y_train):
    entropyClassifier=DecisionTreeClassifier(criterion = "entropy")
    entropyClassifier.fit(x_train,y_train)
    return entropyClassifier

def predict(x_test,classifier,objectiveCol):
    y_pred=classifier.predict(x_test)
    print("predicted values for "+objectiveCol)
    print(y_pred)
    return y_pred

def evaluation(y_test,y_pred):
    print("Confusion matrix: ",confusion_matrix(y_test,y_pred))
    print("accuracy: ",accuracy_score(y_test,y_pred))
    print("Report: ",classification_report(y_test,y_pred))

def printMatrix(c,matrix,classifier):
    plt.clf()
    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plot colorbar to the right
    plt.colorbar()
    fmt = 'd'
    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    classes = ['0', '1', '2', '3', '4', '5', '6']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label', size=15)
    plt.savefig("TreeConfusionMatrix/" + c+'_'+classifier+'.png', dpi=150)
    plt.close()

def main():
    objectiveCols = ['alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine', 'caffeine',
                     'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legal', 'LSD',
                     'methadone', 'mushrooms', 'nicotine', 'volatileSubstance']
    dataset=importDataset()
    for c in objectiveCols:
        x,y,x_train,x_test,y_train,y_test=splitDataset(dataset,c)
        gini=giniBasedClassifier(x_train,x_test,y_train)
        entropy=entropyBasedClassifier(x_train,x_test,y_train)


        print("results of Gini Index for "+ c)
        yPredGini=predict(x_test,gini,c)
        evaluation(y_test,yPredGini)
        confusionMatrix=confusion_matrix(y_test, yPredGini)
        printMatrix(c,confusionMatrix,'gini')


        print("results of Entropy for " + c)
        yPredEntropy = predict(x_test, entropy, c)
        evaluation(y_test, yPredEntropy)
        confusionMatrix = confusion_matrix(y_test, yPredEntropy)
        printMatrix(c, confusionMatrix, 'entropy')

if __name__=="__main__":
    main()


