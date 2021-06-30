

import itertools
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
from sdv.tabular import GaussianCopula, CopulaGAN
from sdv.tabular import CTGAN

import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn import tree


""" Define the output directory for Plots (global) """
plot_dir = 'Plots'
if not os.path.isdir(plot_dir):
    os.system('mkdir ' + plot_dir )




""" Dictionary containing the training and prediction features """
features = { 'adult' :
                 {'features':  ["age","workclass","fnlwgt","education","education-num",
                                "marital-status","occupation","relationship","race","sex",
                                "capital-gain","capital-loss","hours-per-week","native-country"],
                  'target': ["class"] }
             }






def splitDataset(dataset = '', train_features = [], target_features = [] ):
    """ Split the dataset provided into train and test """


    train_set = dataset[train_features]
    target_set = dataset[target_features]

    x_train, x_test, y_train, y_test = train_test_split(train_set, target_set,
                                                        test_size=0.30,
                                                        shuffle=True)
    # first shuffle the dataset !!!!
    # then train_test_split

    return train_set, target_set, x_train, x_test, y_train, y_test


def Classifier(x_train,y_train):
    cl=RandomForestClassifier();
    cl.fit(x_train, y_train)
    return cl


def predict(x_test, classifier):
    """ Wraper for the sklearn predict function """
    y_pred=classifier.predict(x_test)
    return y_pred

def evaluation(y_test,y_pred, normalize='true'):
    confusion_m = confusion_matrix(y_test, y_pred, normalize=normalize)
    accuracy = accuracy_score(y_test,y_pred)

    # return the class_report as a dictionary
    report = classification_report(y_test, y_pred, output_dict= True)
    return confusion_m, accuracy, report


def splitAndTrainOriginalModel():
    print('classification of adult income dataset')
    dataset = pd.read_csv("input_data/" + 'adult_cleaned.csv')
    x, y, x_train, x_test, y_train, y_test = splitDataset(dataset=dataset, train_features=
    ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation"
        , "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"],
                                                          target_features='class')
    x_train.to_csv('splittedDatasets/x_trainAdult.csv', index=False)
    x_test.to_csv('splittedDatasets/x_testAdult.csv', index=False)
    y_train.to_csv('splittedDatasets/y_trainAdult.csv', index=False)
    y_test.to_csv('splittedDatasets/y_testAdult.csv', index=False)
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('classification of titanic survivors dataset')
    dataset = pd.read_csv("input_data/" + 'titanic_cleaned.csv')
    x, y, x_train, x_test, y_train, y_test = splitDataset(dataset=dataset, train_features=
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], target_features=['Survived'])
    x_train.to_csv('splittedDatasets/x_trainTitanic.csv', index=False)
    x_test.to_csv('splittedDatasets/x_testTitanic.csv', index=False)
    y_train.to_csv('splittedDatasets/y_trainTitanic.csv', index=False)
    y_test.to_csv('splittedDatasets/y_testTitanic.csv', index=False)
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('classification of ads')
    dataset = pd.read_csv("input_data/" + 'ads_cleaned.csv')
    x, y, x_train, x_test, y_train, y_test = splitDataset(dataset=dataset, train_features=
    ['Gender', 'Age', 'EstimatedSalary'], target_features=['Purchased'])
    x_train.to_csv('splittedDatasets/x_trainAds.csv', index=False)
    x_test.to_csv('splittedDatasets/x_testAds.csv', index=False)
    y_train.to_csv('splittedDatasets/y_trainAds.csv', index=False)
    y_test.to_csv('splittedDatasets/y_testAds.csv', index=False)
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

def generateGaussianCopulaModel():
    #I need to merge together x_train y_train
    #generate the new dataset with gaussian copula
    #split again the new data
    #fit the classifier
    #test the classifier using x_test and y_test
    print('generating artificial data for adult income dataset using gaussian model')
    x_train = pd.read_csv('splittedDatasets/x_trainAdult.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAdult.csv')
    model = GaussianCopula()
    model.fit(pd.merge(x_train,y_train,left_index=True,right_index=True))
    newAdultIncomeData = model.sample(5000)
    newAdultIncomeData['class'].to_csv('generatedData/y_trainAdultGaussian.csv', index=False)
    del newAdultIncomeData['class']
    newAdultIncomeData.to_csv('generatedData/x_trainAdultGaussian.csv', index=False)
    x_test=pd.read_csv('splittedDatasets/x_testAdult.csv')
    y_test = pd.read_csv('splittedDatasets/y_testAdult.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('generating artificial data for Titanic dataset using gaussian model')
    x_train = pd.read_csv('splittedDatasets/x_trainTitanic.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainTitanic.csv')
    model = GaussianCopula()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(5000)
    newAdultIncomeData['Survived'].to_csv('generatedData/y_trainTitanicGaussian.csv', index=False)
    del newAdultIncomeData['Survived']
    newAdultIncomeData.to_csv('generatedData/x_trainTitanicGaussian.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testTitanic.csv')
    y_test = pd.read_csv('splittedDatasets/y_testTitanic.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('generating artificial data for Ads dataset using gaussian model')
    x_train = pd.read_csv('splittedDatasets/x_trainAds.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAds.csv')
    model = GaussianCopula()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(5000)
    newAdultIncomeData['Purchased'].to_csv('generatedData/y_trainAdsGaussian.csv', index=False)
    del newAdultIncomeData['Purchased']
    newAdultIncomeData.to_csv('generatedData/x_trainAdsGaussian.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testAds.csv')
    y_test = pd.read_csv('splittedDatasets/y_testAds.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

def generateCTGANModel():
    print('generating artificial data for adult income dataset using CTGAN model')
    x_train = pd.read_csv('splittedDatasets/x_trainAdult.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAdult.csv')
    model = CTGAN()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(100)
    newAdultIncomeData['class'].to_csv('generatedData/y_trainAdultCTGAN.csv', index=False)
    del newAdultIncomeData['class']
    newAdultIncomeData.to_csv('generatedData/x_trainAdultCTGAN.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testAdult.csv')
    y_test = pd.read_csv('splittedDatasets/y_testAdult.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('generating artificial data for Titanic dataset using CTGAN model')
    x_train = pd.read_csv('splittedDatasets/x_trainTitanic.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainTitanic.csv')
    model = CTGAN()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(100)
    newAdultIncomeData['Survived'].to_csv('generatedData/y_trainTitanicCTGAN.csv', index=False)
    del newAdultIncomeData['Survived']
    newAdultIncomeData.to_csv('generatedData/x_trainTitanicCTGAN.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testTitanic.csv')
    y_test = pd.read_csv('splittedDatasets/y_testTitanic.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('generating artificial data for Ads dataset using CTGAN model')
    x_train = pd.read_csv('splittedDatasets/x_trainAds.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAds.csv')
    model = CTGAN()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(100)
    newAdultIncomeData['Purchased'].to_csv('generatedData/y_trainAdsCTGAN.csv', index=False)
    del newAdultIncomeData['Purchased']
    newAdultIncomeData.to_csv('generatedData/x_trainAdsCTGAN.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testAds.csv')
    y_test = pd.read_csv('splittedDatasets/y_testAds.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

def generateGTGACopulaModel():
    print('generating artificial data for adult income dataset using CTGAN Copula model')
    x_train = pd.read_csv('splittedDatasets/x_trainAdult.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAdult.csv')
    model = CopulaGAN()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(1000)
    newAdultIncomeData['class'].to_csv('generatedData/y_trainAdultCTGANCopula.csv', index=False)
    del newAdultIncomeData['class']
    newAdultIncomeData.to_csv('generatedData/x_trainAdultCTGANCopula.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testAdult.csv')
    y_test = pd.read_csv('splittedDatasets/y_testAdult.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('generating artificial data for Titanic dataset using CTGAN Copula model')
    x_train = pd.read_csv('splittedDatasets/x_trainTitanic.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainTitanic.csv')
    model = CopulaGAN()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(1000)
    newAdultIncomeData['Survived'].to_csv('generatedData/y_trainTitanicCTGANCopula.csv', index=False)
    del newAdultIncomeData['Survived']
    newAdultIncomeData.to_csv('generatedData/x_trainTitanicCTGANCopula.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testTitanic.csv')
    y_test = pd.read_csv('splittedDatasets/y_testTitanic.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])

    print('generating artificial data for Ads dataset using CTGAN Copula model')
    x_train = pd.read_csv('splittedDatasets/x_trainAds.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAds.csv')
    model = CopulaGAN()
    model.fit(pd.merge(x_train, y_train, left_index=True, right_index=True))
    newAdultIncomeData = model.sample(1000)
    newAdultIncomeData['Purchased'].to_csv('generatedData/y_trainAdsCTGANCopula.csv', index=False)
    del newAdultIncomeData['Purchased']
    newAdultIncomeData.to_csv('generatedData/x_trainAdsCTGANCopula.csv', index=False)
    x_test = pd.read_csv('splittedDatasets/x_testAds.csv')
    y_test = pd.read_csv('splittedDatasets/y_testAds.csv')
    print('fitting and testing RF using the new generated data')
    cf = Classifier(x_train, y_train)
    y_prediction = predict(x_test, cf, )
    confusion_m, accuracy, report = evaluation(y_test, y_prediction)
    print(confusion_m)
    print("Accuracy:", accuracy)
    print("Precision (macro avg):", report['macro avg']['precision'])
    print("Recall (macro avg):", report['macro avg']['recall'])
    print("F1-score (macro avg):", report['macro avg']['f1-score'])















def main():
    warnings.simplefilter('ignore')
    splitAndTrainOriginalModel()
    generateGaussianCopulaModel()
    generateCTGANModel()
    generateGTGACopulaModel()















def plot_line_results(results, dataset):

    if not os.path.isdir('Plots/comparison/'):
        os.mkdir('Plots/comparison/')

    fs = 15

    to_plot = {}

    for cl in ['adult']:
        to_plot[cl] = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        for i in ['recall','f1','precision']:
            print(dataset,' ',cl, ' ', i )
            to_plot[cl][i].append(results[dataset][cl][i])

    for i in ['recall','f1','precision']:

        fig, ax = plt.subplots(figsize = (12,5))

        plt.scatter(range(1,len(to_plot['adult'][i])+1), to_plot['adult'][i] )
        plt.plot(range(1,len(to_plot['adult'][i])+1), to_plot['adult'][i], label = 'adult' )


        plt.ylim(0,0.4)
        ax.set_xticks(range(1,len(to_plot['adult'][i])+1))
        ax.set_xticklabels(rotation=35, fontsize=10 )

        plt.title( i + ' for ' + dataset + ' dataset ', fontsize = fs )
        plt.legend()
        plt.tight_layout()
        plt.grid(ls=':', color='lightgray')
        plt.savefig('Plots/comparison/'+ i + '_comparison_'+ dataset + '.png', dpi = 200)
        plt.close()

if __name__=="__main__":
    main()




