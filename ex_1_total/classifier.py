import itertools
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#import graphviz
from sklearn import tree


""" Define the output directory for Plots (global) """
plot_dir = 'Plots'
if not os.path.isdir(plot_dir):
    os.system('mkdir ' + plot_dir )




""" Dictionary containing the training and prediction features """
features = { 'drugs' :
                 {'features':  ['age', 'gender', 'education', 'ethnicity', 'Nscore',
                                    'Escore', 'Oscore', 'Ascore',
                                    'Cscore', 'Impulsive', 'SS'] ,

                  'target': ['alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine', 'caffeine',
                                    'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy', 'heroin',
                                    'ketamine', 'legal', 'LSD',
                                    'methadone', 'mushrooms', 'nicotine', 'volatileSubstance'] } ,

             'asteroids' :
                 {'features': ['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)',
                               'Relative Velocity km per sec', 'Miss Dist.(kilometers)',
                               'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
                               'Eccentricity', 'Semi Major Axis', 'Inclination', 'Asc Node Longitude',
                               'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
                               'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion'] ,

                    'target':  ['Hazardous'] } ,

             'advertisingBidding':
                 {'features': ['Region', 'City', 'AdExchange', 'Adslotwidth', 'Adslotheight', 'Adslotfloorprice',
                           'CreativeID', 'Biddingprice', 'AdvertiserID',
                           'interest_news', 'interest_eduation', 'interest_automobile',
                           'interest_realestate', 'interest_IT', 'interest_electronicgame',
                           'interest_fashion', 'interest_entertainment', 'interest_luxury',
                           'interest_homeandlifestyle', 'interest_health', 'interest_food',
                           'interest_divine', 'interest_motherhood_parenting', 'interest_sports',
                           'interest_travel_outdoors', 'interest_social', 'Inmarket_3cproduct',
                           'Inmarket_appliances', 'Inmarket_clothing_shoes_bags',
                           'Inmarket_Beauty_PersonalCare', 'Inmarket_infant_momproducts',
                           'Inmarket_sportsitem', 'Inmarket_outdoor', 'Inmarket_healthcareproducts',
                           'Inmarket_luxury', 'Inmarket_realestate', 'Inmarket_automobile',
                           'Inmarket_finance', 'Inmarket_travel', 'Inmarket_education',
                           'Inmarket_service', 'interest_art_photography_design',
                           'interest_onlineliterature', 'Inmarket_electronicgame', 'interest_3c',
                           'Inmarket_book', 'Inmarket_medicine', 'Inmarket_food_drink',
                           'interest_culture', 'interest_sex', 'Demographic_gender_male',
                           'Demographic_gender_famale', 'Inmarket_homeimprovement', 'Payingprice',
                           'imp', 'click', 'Browser', 'Adslotvisibility', 'Adslotformat'],
                   'target': ['conv']} ,


            'breastCancer' :
                 {'features':  ['radiusMean', ' textureMean', ' perimeterMean', ' areaMean',
                                ' smoothnessMean', ' compactnessMean', ' concavityMean',
                                ' concavePointsMean', ' symmetryMean', ' fractalDimensionMean',
                                ' radiusStdErr', ' textureStdErr' ,' perimeterStdErr' ,' areaStdErr',
                                ' smoothnessStdErr', ' compactnessStdErr', ' concavityStdErr',
                                ' concavePointsStdErr' ,' symmetryStdErr' ,' fractalDimensionStdErr',
                                ' radiusWorst', ' textureWorst' ,' perimeterWorst' ,' areaWorst',
                                ' smoothnessWorst' ,' compactnessWorst' ,' concavityWorst',
                                ' concavePointsWorst', ' symmetryWorst', ' fractalDimensionWorst'],
                  'target': ['class'] }


             }



def importDataset(ds, full_chain = True):
    """ Directory with the datasets paths """
    datasets = {'drugs'     : "input_data/drug_consumption.data_cleaned.csv" ,

                'asteroids' : "input_data/asteroids_cleaned.csv" ,

                'breastCancer':
                    {'learn' : "input_data/breast-cancer-diagnostic.shuf.lrn_cleaned.csv",
                     'test_x': "input_data/breast-cancer-diagnostic.shuf.tes_cleaned.csv",
                     'test_y': "input_data/breast-cancer-diagnostic.shuf.sol.ex_cleaned.csv" },

                'advertisingBidding':
                    {'learn' : "input_data/advertisingBidding.shuf.lrn_cleaned.csv",
                     'test_x': "input_data/advertisingBidding.shuf.tes_cleaned.csv",
                     'test_y': "input_data/advertisingBidding.shuf.sol.ex_cleaned.csv"},
                }

    """ Return a pandas dataframe """
    if full_chain:
        if ds in ['asteroids' , 'drugs']:
            dataset = pd.read_csv( datasets[ds])
        else:
            dataset = pd.read_csv(datasets[ds]['learn'])
        return dataset

    else:
        train = pd.read_csv( datasets[ds]['learn'])

        train_x = train[ features[ds]['features'] ]
        train_y = train[ features[ds]['target'] ]

        test_x = pd.read_csv( datasets[ds]['test_x'])
        test_y = pd.read_csv( datasets[ds]['test_y'])

        return train_x, train_y, test_x, test_y



def splitDataset(dataset = '', train_features = [], target_features = [] ):
    """ Split the dataset provided into train and test """

    train_set  = dataset[train_features]
    target_set = dataset[target_features]

    x_train, x_test, y_train, y_test = train_test_split(train_set, target_set,
                                                        test_size=0.30,
                                                        shuffle = True)
    # first shuffle the dataset !!!!
    # then train_test_split

    return train_set, target_set, x_train, x_test, y_train, y_test


def Classifier(x_train,y_train, classifier='tree', criterion="gini",  n_neighbors=4):
    """ Run a DecisionTreeClassifier algorithm, with the specified criterion
    input  ::   classifier:  tree   DecisionTreeClassifier
                             kNN    KNeighborsClassifier
                             naive  GaussianNB
                criterion: gini
                           entropy     valid only for DecisionTreeClassifier
                n_neighbors : (4)      valid only for KNeighborsClassifier
    """

    if classifier == 'DecisionTree':
        cl = DecisionTreeClassifier(criterion = criterion )
    if classifier == 'KNeighbors':
        cl = KNeighborsClassifier(n_neighbors= n_neighbors )
    if classifier == 'GaussianNB':
        cl = GaussianNB()
    cl.fit(x_train, y_train)
    return cl


def predict(x_test, classifier, objectiveCol):
    y_pred=classifier.predict(x_test)
    #print("predicted values for "+objectiveCol)
    #print(y_pred)
    return y_pred

def evaluation(y_test,y_pred, normalize='true'):
    confusion_m = confusion_matrix(y_test, y_pred, normalize=normalize)
    accuracy = accuracy_score(y_test,y_pred)

    # return the class_report as a dictionary
    report = classification_report(y_test, y_pred, output_dict= True)
    #print("Confusion matrix: ", confusion_m)
    #print("accuracy: ", accuracy)
    #for i in report:
    #    print(i, ":", report[i])
    #print("Report: ", report)

    return confusion_m, accuracy, report

def printMatrix(target, matrix, classifier, param, dataset, balance = '', validation = ''):
    """ Creates a confusion matrix """
    fs = 12
    plt.figure(figsize=(6,5))
    # place labels at the top

    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin = 0, vmax = 1)
    # plot colorbar to the right
    plt.colorbar()
    fmt = 'd'
    fmt = 'f.2'
    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        # if background is dark, use a white number, and vice-versa
        #plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center",
        #         color="white" if matrix[i, j] > thresh else "black")
        plt.text(j, i, '{0:.2f}'.format(matrix[i, j]) , horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    ticks = { 'asteroids' : ['Non Hazardous' , 'Hazardous'],
              'drugs': ["Never", ">10 Years Ago", "Last Decade", "Last Year", "Last Month",
              "Last Week", "Last Day"],
              'breastCancer' :  ['Reccurence' , 'No recurrence'] ,
              'advertisingBidding' : ['Buy','Not Buy']
              }

    # "recurrence-events" or not ("no-recurrence-events")
    #
    classes = ticks[dataset]

    classes = range(0,len(matrix))
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, ticks[dataset], rotation = 20,  fontsize = 8)
    plt.yticks(tick_marks, ticks[dataset], rotation = 20 , va="center", fontsize = 8)

    titles = { 'drugs' : target ,
               'asteroids': 'Asteroids',
               'breastCancer': 'Breast Cancer',
               'advertisingBidding': 'Advertising Bidding'}

    plt.title(str(classifier) + '[' + str(param) + ']' +
              ' Confusion Matrix - ' +validation+ ' ('+titles[dataset]+')', y=-0.2, fontsize=fs-1)

    plt.ylabel('True label', size=fs)
    plt.xlabel('Predicted label', size=fs)

    # Save the fig
    out = plot_dir + '/ConfusionMatrix/' + dataset + '/'

    if not os.path.isdir(plot_dir + '/' + 'ConfusionMatrix'):
        os.system('mkdir ' + plot_dir + '/' + 'ConfusionMatrix' )

    if not os.path.isdir(out):
        os.system('mkdir ' + out )
    plt.tight_layout()

    plt.savefig(out + dataset + '_' + str(classifier) + '_' +
                str(param) + '_balance_' + str(balance) + '_' + validation + '.png', dpi=150)

    plt.close()



def plot_balance_ds(df, dataset):
    """ make a plot of the different classes for the target attributes
    (frequency of drug usage for drugs,
    binary True/False for the others """

    fs = 17

    ticks = { 'asteroids' : ['Hazardous' , 'Non Hazardous'],
              'drugs': ["Never", ">10 Years Ago", "Last Decade", "Last Year", "Last Month",
              "Last Week", "Last Day"],
              'breastCancer' :  ['Recurrence events' , 'No recurrence events'] ,
              'advertisingBidding' : ['Buy','Not Buy']
              }

    Labels = ticks[dataset]

    if dataset == 'drugs':
        drugs = features[dataset]['target']
        classes = range(0,7)  # for drugs
        res = {}

        for c in classes:
            res[c] = []
            for d in drugs:
                a = len( np.where(df[d] == c )[0] )
                res[c].append(a)

        fig, ax = plt.subplots(figsize = (12,7))
        ax.set_title('Class Distribution for ' + dataset, fontsize=fs, y=1.03)

        width = 0.7
        cum_sum = np.full( len(res[0]), 0)

        for i in range(0,7):
             ax.bar (drugs, res[i], width, label=Labels[i], bottom = cum_sum)
             cum_sum += res[i]


        ax.set_xticklabels(drugs, rotation=35, fontsize=10 )
        ax.legend()
        ax.set_ylabel('Counts', fontsize=fs)

        plt.tight_layout()
        plt.grid(ls=':', color='lightgray')
        plt.savefig('Plots/Inbalance_' + dataset + '.png', dpi=200)

    else:
        width = 0.7
        data = df[features[dataset]['target']]
        f = np.count_nonzero(df[features[dataset]['target']] == False )
        t = np.count_nonzero(df[features[dataset]['target']] == True )

        fig, ax = plt.subplots(figsize = (6,7))

        ax.bar( ticks[dataset], [t,f] , width )
        ax.set_xticklabels(ticks[dataset], fontsize=fs )

    ax.set_title('Class Distribution for ' + dataset, fontsize=fs, y=1.03)
    ax.legend()

    ax.set_ylabel('Counts', fontsize=fs)
    plt.tight_layout()
    plt.grid(ls=':', color='lightgray')
    plt.savefig('Plots/Inbalance_' + dataset +'.png', dpi=200)

    print('*** Plotted inbalance distributions drugs ***')




def plot_reports(report_summary, classifier, dataset):

    fs = 15
    labels = report_summary[0].keys()  # class labels

    os.system('mkdir Plots/validation/')
    for feature,num in zip( features[dataset]['target'], range(len(features[dataset]['target'])) ):
        dic = report_summary[num]
        precision, recall, f1= [], [], []

        classes = range(0,7)
        for l in classes:
            lab = str(l)
            precision.append(dic[lab]['precision'])
            recall.append(dic[lab]['recall'])
            f1.append(dic[lab]['f1-score'])

        plt.plot(classes, precision, label = 'Precision')
        plt.plot(classes, recall, label = 'Recall')
        plt.plot(classes, f1, label='f1-score')
        plt.ylim(0,1)
        plt.legend(loc = 'upper left')
        plt.grid(ls=':', color='lightgray')
        plt.title(classifier + ' Measures - ' + feature, fontsize=fs, y=1.02 )
        plt.tight_layout()
        print("*** Done plotting *** ", feature )
        plt.savefig('Plots/validation/' + dataset + '/' + feature + '.png', dpi = 200)
        plt.close()


def balance_ds(ds, dataset):
    """ Return a dataset with balanced classes.
        Select the minority class, end extracts its length L.
        Then select L + 10% random entries from the majority list. """

    target = features[dataset]['target'][0]
    if dataset != 'drugs':
        classes = ['True', 'False']

        t = ds.loc[ds[target] == True] # always minority
        length = int( len(t) + 0.1* len(t) )

        f = ds.loc[ds[target] == False]
        f = f.sample(frac=1).reset_index() # shuffling the majority class. Resetting indices for accessing it below

    randomlist = []
    for i in range(0, length):
        n = random.randint(1,len(f)-1)
        randomlist.append(n)

    random_f = f.iloc[randomlist]

    full_ds = pd.concat([t, random_f])
    return full_ds.sample(frac=1)

def clean_fast():
    ds = pd.read_csv('input_data/advertisingBidding.shuf.lrn.csv').dropna()
    for rem in ['RowID', 'UserID', 'BidID', 'IP', 'Domain', 'URL', 'Time_Bid', 'AdslotID']:
        del ds[rem]
    labelencoder = LabelEncoder()
    for cl in ['Browser', 'Adslotvisibility', 'Adslotformat']:
        # cleanedTest[cl] = labelencoder.fit_transform(cleanedTest[cl].astype(str))
        ds[cl] = labelencoder.fit_transform(ds[cl].astype(str))

    return ds


###################################
# Main input parameters to choose #
###################################

""" Choose validation type [string]
    validation = "holdout", "crossvalidation"
    For cross validation, choose the number of splits [int]
    cv_splits = 10 (default) """

validation = 'crossvalidation'
cv_splits = 10

""" Choose the types of classifier [list]
    classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB'] """
classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB']

""" Choose the datasets [list]
    datasets = ['asteroids','advertisingBidding' , 'breastCancer', 'drugs' ] """
datasets = ['asteroids','advertisingBidding' , 'breastCancer', 'drugs' ]
datasets = ['asteroids' ]

""" Choose if the target features of the datasets should be balanced before classification
    balance = True,False """
balance = True

# to do ?
train_test = True


def main():

    for dataset in datasets:
        # Loading the data frames
        if dataset in ['asteroids', 'drugs']:
            ds = importDataset(dataset)
        else:
            ds = importDataset(dataset, full_chain= True)

            # need to implement the separate testing only
            # xtrain, ytrain, xtest, ytest = importDataset(dataset)

        # reshuffle data
        ds = ds.sample(frac=1).reset_index()

        a = plot_balance_ds(ds, dataset)

        # balance the dataset if balance = True
        if balance and dataset != 'drugs':
             ds = balance_ds(ds, dataset)

        for target in features[dataset]['target']:

            # In case of cv the x and y will be used and the rest overwritten
            if dataset == 'drugs':
                x, y, x_train, x_test, y_train, y_test = splitDataset(dataset= ds, train_features= features[dataset]['features'],
                                                                      target_features= target )
            else:
                x, y, x_train, x_test, y_train, y_test = splitDataset(dataset=ds, train_features=features[dataset]['features'],
                                                                      target_features=features[dataset]['target'])
            if validation == 'holdout' :  # must split the data into train-test
                # Simple Hold Out

                print('*** I Split dataset ' , dataset , ' ***')

                for classifier in classifiers:
                    if classifier == 'DecisionTree' : # Run DecisionTreeClassifier
                        for param in ['gini','entropy'] :  # run the classifier with two different parameter
                            #for param in ['gini', 'entropy']:  # run the classifier with two different parameter
                            print("\n\nResults of " + classifier + " " + param + ". Index for " + target )
                            cf = Classifier(x_train,y_train, classifier=classifier, criterion=param )

                            y_prediction=predict(x_test, cf, target)
                            confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                            print("Accuracy:", accuracy)
                            print("Precision (macro avg):", report['macro avg']['precision'])
                            print("Recall (macro avg):", report['macro avg']['recall'])
                            print("F1-score (macro avg):", report['macro avg']['f1-score'])
                            printMatrix(target, confusion_m, classifier, param, dataset,
                                        balance = balance, validation = validation)

                    if classifier == 'KNeighbors':
                        for param in [5, 10 , 50]:
                            print("\n\nResults of " + classifier + " with k=" + str(param) + ". Index for " + target )
                            cf = Classifier(x_train, y_train, classifier=classifier, n_neighbors = param )

                            y_prediction=predict(x_test, cf, target)
                            confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                            print("Accuracy:", accuracy)
                            print("Precision (macro avg):", report['macro avg']['precision'])
                            print("Recall (macro avg):", report['macro avg']['recall'])
                            print("F1-score (macro avg):", report['macro avg']['f1-score'])
                            printMatrix(target, confusion_m, classifier, param, dataset,
                                        balance = balance, validation = validation)

                    if classifier == 'GaussianNB':
                        for param in ['naiveB']:
                            print("\n\nResults of " + classifier + " " + param + ". Index for " + target )
                            cf = Classifier(x_train, y_train, classifier=classifier)

                            y_prediction=predict(x_test, cf, target)
                            confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                            print("Accuracy:", accuracy)
                            print("Precision (macro avg):", report['macro avg']['precision'])
                            print("Recall (macro avg):", report['macro avg']['recall'])
                            print("F1-score (macro avg):", report['macro avg']['f1-score'])
                            printMatrix(target, confusion_m, classifier, param, dataset,
                                        balance = balance, validation = validation)

            elif validation == 'crossvalidation':
                if dataset == "drugs":  # will not run cross validation on drug dataset
                    continue

                confusion_m = np.zeros((2,2))
                accuracy = 0
                macro_precision = 0
                macro_recall = 0
                macro_f1 = 0

                kf = KFold(n_splits=cv_splits)
                for classifier in classifiers:
                    if classifier == 'DecisionTree' : # Run DecisionTreeClassifier
                        for param in ['gini','entropy'] :  # run the classifier with two different parameter
                            confusion_m = np.zeros((2,2))
                            accuracy = 0
                            macro_precision = 0
                            macro_recall = 0
                            macro_f1 = 0
                            print("\n\n")
                            for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]
                                #for param in ['gini', 'entropy']:  # run the classifier with two different parameter
                                print("Results of " + classifier + " " + param + ". Index for " + target )
                                cf = Classifier(x_train,y_train, classifier=classifier, criterion=param )
                                y_prediction=predict(x_test,cf,target)
                                confusion_mT, accuracyT, report = evaluation(y_test, y_prediction, None)
                                confusion_m += confusion_mT
                                accuracy += accuracyT
                                macro_precision += report['macro avg']['precision']
                                macro_recall += report['macro avg']['recall']
                                macro_f1 += report['macro avg']['f1-score']
                            # after all folds, rescale the matrix and the accuracy and print it
                            row_sums = confusion_m.sum(axis=1)
                            confusion_m = confusion_m / row_sums[:, np.newaxis]
                            accuracy /= cv_splits
                            macro_precision /= cv_splits
                            macro_recall /= cv_splits
                            macro_f1 /= cv_splits
                            print("Accuracy:", accuracy)
                            print("Precision (macro avg):", macro_precision)
                            print("Recall (macro avg):", macro_recall)
                            print("F1-score (macro avg):", macro_f1)
                            printMatrix(target, confusion_m, classifier, param, dataset,
                                        balance = balance, validation=validation)

                    if classifier == 'KNeighbors':
                        for param in [5, 10 , 50]:
                            confusion_m = np.zeros((2,2))
                            accuracy = 0
                            macro_precision = 0
                            macro_recall = 0
                            macro_f1 = 0
                            print("\n\n")
                            for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]
                                print("Results of " + classifier + " with k=" + str(param) + ". Index for " + target )
                                cf = Classifier(x_train, y_train, classifier=classifier, n_neighbors = param )
                                y_prediction=predict(x_test,cf,target)
                                confusion_mT, accuracyT, report = evaluation(y_test, y_prediction, None)
                                confusion_m += confusion_mT
                                accuracy += accuracyT
                                macro_precision += report['macro avg']['precision']
                                macro_recall += report['macro avg']['recall']
                                macro_f1 += report['macro avg']['f1-score']
                            # after all folds, rescale the matrix and the accuracy and print it
                            row_sums = confusion_m.sum(axis=1)
                            confusion_m = confusion_m / row_sums[:, np.newaxis]
                            accuracy /= cv_splits
                            macro_precision /= cv_splits
                            macro_recall /= cv_splits
                            macro_f1 /= cv_splits
                            print("Accuracy:", accuracy)
                            print("Precision (macro avg):", macro_precision)
                            print("Recall (macro avg):", macro_recall)
                            print("F1-score (macro avg):", macro_f1)
                            printMatrix(target, confusion_m, classifier, param, dataset,
                                        balance=balance, validation=validation)

                    if classifier == 'GaussianNB':
                        for param in ['naiveB']:
                            confusion_m = np.zeros((2,2))
                            accuracy = 0
                            macro_precision = 0
                            macro_recall = 0
                            macro_f1 = 0
                            print("\n\n")
                            for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]
                                print("Results of " + classifier + " " + param + ". Index for " + target )
                                cf = Classifier(x_train, y_train, classifier=classifier)
                                y_prediction=predict(x_test, cf, target)
                                confusion_mT, accuracyT, report = evaluation(y_test, y_prediction, None )
                                confusion_m += confusion_mT
                                accuracy += accuracyT
                                macro_precision += report['macro avg']['precision']
                                macro_recall += report['macro avg']['recall']
                                macro_f1 += report['macro avg']['f1-score']
                            # after all folds, rescale the matrix and the accuracy and print it
                            row_sums = confusion_m.sum(axis=1)
                            confusion_m = confusion_m / row_sums[:, np.newaxis]
                            accuracy /= cv_splits
                            macro_precision /= cv_splits
                            macro_recall /= cv_splits
                            macro_f1 /= cv_splits
                            print("Accuracy:", accuracy)
                            print("Precision (macro avg):", macro_precision)
                            print("Recall (macro avg):", macro_recall)
                            print("F1-score (macro avg):", macro_f1)
                            printMatrix(target, confusion_m, classifier, param, dataset,
                                        balance = balance, validation = validation)
            else:
                # TO DO must only test with the given dataset, cancer and bidding
                print(0)


        #dummy = plot_reports(report_summary, classifier, dataset)

if __name__=="__main__":

    classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB']
    datasets = ['asteroids', 'advertisingBidding', 'breastCancer', 'drugs']

    balance = True
    validation = 'holdout'
    main()

    balance = False
    validation = 'holdout'
    main()

    balance = True
    validation = 'crossvalidation'
    main()

    balance = False
    validation = 'crossvalidation'
    main()
