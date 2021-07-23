

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
from sdv import *

import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn import tree


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
    printConfusionMatrix(confusion_m, 'AdultIncomeCopula', 'adult')

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
    printConfusionMatrix(confusion_m, 'titanicCopula', 'titanic')

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
    printConfusionMatrix(confusion_m, 'AdsCopula', 'ads')

def generateCTGANModel():
    print('generating artificial data for adult income dataset using CTGAN model')
    x_train = pd.read_csv('splittedDatasets/x_trainAdult.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAdult.csv')
    model = CTGAN()
    dataset=pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit(dataset[0:1500])
    newAdultIncomeData = model.sample(1500)
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
    printConfusionMatrix(confusion_m, 'AdultIncomeCTGAN', 'adult')

    print('generating artificial data for Titanic dataset using CTGAN model')
    x_train = pd.read_csv('splittedDatasets/x_trainTitanic.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainTitanic.csv')
    model = CTGAN()
    dataset = pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit(dataset[0:1500])
    newTitanicIncomeData = model.sample(1500)
    newTitanicIncomeData['Survived'].to_csv('generatedData/y_trainTitanicCTGAN.csv', index=False)
    del newTitanicIncomeData['Survived']
    newTitanicIncomeData.to_csv('generatedData/x_trainTitanicCTGAN.csv', index=False)
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
    printConfusionMatrix(confusion_m, 'titanicCTGAN', 'titanic')

    print('generating artificial data for Ads dataset using CTGAN model')
    x_train = pd.read_csv('splittedDatasets/x_trainAds.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAds.csv')
    model = CTGAN()
    dataset = pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit(dataset[0:1500])
    newAdsData = model.sample(1500)
    newAdsData['Purchased'].to_csv('generatedData/y_trainAdsCTGAN.csv', index=False)
    del newAdsData['Purchased']
    newAdsData.to_csv('generatedData/x_trainAdsCTGAN.csv', index=False)
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
    printConfusionMatrix(confusion_m, 'adsCTGAN', 'ads')


def generateGTGACopulaModel():
    print('generating artificial data for adult income dataset using CTGAN Copula model')
    x_train = pd.read_csv('splittedDatasets/x_trainAdult.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAdult.csv')
    model = CopulaGAN()
    dataset = pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit(dataset[0:1500])
    newAdultIncomeData = model.sample(1500)
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
    printConfusionMatrix(confusion_m, 'adultIncomeCTGANCopula', 'adult')

    print('generating artificial data for Titanic dataset using CTGAN Copula model')
    x_train = pd.read_csv('splittedDatasets/x_trainTitanic.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainTitanic.csv')
    model = CopulaGAN()
    dataset = pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit(dataset[0:1500])
    newTitanicData = model.sample(1500)
    newTitanicData['Survived'].to_csv('generatedData/y_trainTitanicCTGANCopula.csv', index=False)
    del newTitanicData['Survived']
    newTitanicData.to_csv('generatedData/x_trainTitanicCTGANCopula.csv', index=False)
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
    printConfusionMatrix(confusion_m, 'titanicCTGANCopula', 'titanic')

    print('generating artificial data for Ads dataset using CTGAN Copula model')
    x_train = pd.read_csv('splittedDatasets/x_trainAds.csv')
    y_train = pd.read_csv('splittedDatasets/y_trainAds.csv')
    model = CopulaGAN()
    dataset = pd.merge(x_train, y_train, left_index=True, right_index=True)
    model.fit(dataset[0:1500])
    newAdsData = model.sample(1500)
    newAdsData['Purchased'].to_csv('generatedData/y_trainAdsCTGANCopula.csv', index=False)
    del newAdsData['Purchased']
    newAdsData.to_csv('generatedData/x_trainAdsCTGANCopula.csv', index=False)
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
    printConfusionMatrix(confusion_m, 'adsCTGANCopula', 'ads')




