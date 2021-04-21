import itertools
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

""" Define the output directory for Plots (global) """
plot_dir = 'Plots'
if not os.path.isdir(plot_dir):
    os.system('mkdir ' + plot_dir )


def importDataset():
    trainSet= pd.read_csv('data/cleanedTrain.csv', )
    testSet=pd.read_csv('data/cleanedTest.csv', )
    solutionSet=pd.read_csv('data/cleanedSolution.csv',)
    return trainSet,testSet,solutionSet



def splitDataset(dataset = '', train_features = [], target_features = [] ):
    """ Split the dataset provided into train and test """

    train_set  = dataset[train_features]
    target_set = dataset[target_features]

    x_train, x_test, y_train, y_test = train_test_split(train_set, target_set, test_size=0.30)

    return train_set, target_set, x_train, x_test, y_train, y_test


def Classifier(x_train,y_train, classifier = 'tree', criterion = "gini" , n_neighbors = 4  ):
    """ Run a DecisionTreeClassifier algorithm, with the specified criterion
    input  ::   classifier:  tree   DecisionTreeClassifier
                             kNN    KNeighborsClassifier
                             naive  GaussianNB

                criterion: gini
                           entropy     valid only for DecisionTreeClassifier

                n_neighbors : (4)      valid only for KNeighborsClassifier

    """

    if classifier == 'DecisionTree':
        cl=DecisionTreeClassifier(criterion = criterion )
    if classifier == 'KNeighbors':
        cl = KNeighborsClassifier(n_neighbors= n_neighbors )
    if classifier == 'GaussianNB':
        cl = GaussianNB()
    cl.fit(x_train, y_train)
    return cl


def predict(x_test, classifier, objectiveCol):
    y_pred=classifier.predict(x_test)
    print("predicted values for "+objectiveCol)
    print(y_pred)
    return y_pred

def evaluation(y_test,y_pred):
    print("Confusion matrix: ",confusion_matrix(y_test,y_pred))
    print("accuracy: ",accuracy_score(y_test,y_pred))
    print("Report: ",classification_report(y_test,y_pred))

def printMatrix(target, matrix, classifier, param):
    """ Creates a confusion matrix """
    plt.clf()
    # place labels at the top

    #plt.gca().xaxis.tick_top()
    #plt.gca().xaxis.set_label_position('top')
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
    classes = ['0', '1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.title(str(classifier) + ' Confusion Matrix' , y = 1.02 )

    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label', size=15)

    # Save the fig
    out = plot_dir + '/ConfusionMatrix/'
    if not os.path.isdir(out):
        os.system('mkdir ' + out )
    plt.tight_layout()

    plt.savefig(out + target + '_' + str(classifier) +  '_' + str(param) + '.png', dpi=150)
    plt.close()



""" Dictionary containing the training and prediction features """

features = { 'advertisingBidding' : { 'train':  ['Region', 'City', 'AdExchange', 'Adslotwidth', 'Adslotheight','Adslotfloorprice', 'CreativeID' ,'Biddingprice' ,'AdvertiserID',
                                     'interest_news', 'interest_eduation', 'interest_automobile',
                                     'interest_realestate', 'interest_IT', 'interest_electronicgame',
                                     'interest_fashion', 'interest_entertainment', 'interest_luxury',
                                     'interest_homeandlifestyle', 'interest_health', 'interest_food',
                                     'interest_divine', 'interest_motherhood_parenting', 'interest_sports',
                                     'interest_travel_outdoors', 'interest_social', 'Inmarket_3cproduct',
                                     'Inmarket_appliances' ,'Inmarket_clothing_shoes_bags',
                                     'Inmarket_Beauty_PersonalCare', 'Inmarket_infant_momproducts',
                                     'Inmarket_sportsitem', 'Inmarket_outdoor' ,'Inmarket_healthcareproducts',
                                     'Inmarket_luxury' ,'Inmarket_realestate', 'Inmarket_automobile',
                                     'Inmarket_finance', 'Inmarket_travel', 'Inmarket_education',
                                     'Inmarket_service', 'interest_art_photography_design',
                                     'interest_onlineliterature', 'Inmarket_electronicgame', 'interest_3c',
                                     'Inmarket_book', 'Inmarket_medicine', 'Inmarket_food_drink',
                                     'interest_culture' ,'interest_sex', 'Demographic_gender_male',
                                     'Demographic_gender_famale', 'Inmarket_homeimprovement', 'Payingprice',
                                     'imp', 'click', 'Browser', 'Adslotvisibility' ,'Adslotformat'],
                         'target': ['conv'] }


             }





classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB']

def main():

    dataset = 'advertisingBidding'
    trainSet,testSet,solSet = importDataset()
    target='conv'


    for classifier in classifiers:


        for target in features[dataset]['target']:
            x, y, x_train, x_test, y_train, y_test = splitDataset(trainSet,train_features=features[dataset]['train'],target_features=target)


        if classifier == 'DecisionTree' : # Run DecisionTreeClassifier
            for param in ['gini' , 'entropy'] :  # run the classifier with two different parameter
                cf = Classifier(x_train,y_train, classifier=classifier, criterion=param )
                print("results of " + param + " Index for " + target )
                y_prediction=predict(x_test,cf,target)
                evaluation(y_test, y_prediction )
                confusionMatrix=confusion_matrix(y_test, y_prediction)
                printMatrix(target, confusionMatrix, classifier, param)

        if classifier == 'KNeighbors':
            for param in [4]:
                cf = Classifier(x_train,y_train, classifier=classifier, n_neighbors=param )
                print("results of " + str(param) + " Index for " + target )
                y_prediction=predict(x_test,cf,target)
                evaluation(y_test, y_prediction )
                confusionMatrix=confusion_matrix(y_test, y_prediction)
                printMatrix(target, confusionMatrix, classifier, param)

        if classifier == 'GaussianNB':
            for param in ['naiveB']:
                cf = Classifier(x_train, y_train, classifier=classifier)
                print("results of " + param + " Index for " + target)
                y_prediction = predict(x_test, cf, target)
                evaluation(y_test, y_prediction)
                confusionMatrix = confusion_matrix(y_test, y_prediction)
                printMatrix(target, confusionMatrix, classifier, param)



if __name__=="__main__":
    main()
