import itertools
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    confusion_m = confusion_matrix(y_test, y_pred, normalize='true')
    accuracy = accuracy_score(y_test,y_pred)

    # return the class_report as a dictionary
    report = classification_report(y_test, y_pred, output_dict= True)
    print("Confusion matrix: ", confusion_m)
    #print("accuracy: ", accuracy)
    print("Report: ", report)

    return confusion_m, accuracy, report

def printMatrix(target, matrix, classifier, param, dataset):
    """ Creates a confusion matrix """
    fs = 12
    plt.clf()
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

    ticks = { 'asteroids' : ['Hazardous' , 'Non Hazardous'],
              'drugs': ["Never", ">10 Years Ago", "Last Decade", "Last Year", "Last Month",
              "Last Week", "Last Day"],
              'breastCancer' :  ['0' , '1'] ,
              'advertisingBidding' : ['0','1']
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

    plt.title(str(classifier) + ' Confusion Matrix - ' + titles[dataset] , y = -0.2 , fontsize = fs )

    plt.ylabel('True label', size=fs)
    plt.xlabel('Predicted label', size=fs)

    # Save the fig
    out = plot_dir + '/ConfusionMatrix/' + dataset + '/'

    if not os.path.isdir(plot_dir + '/' + 'ConfusionMatrix'):
        os.system('mkdir ' + plot_dir + '/' + 'ConfusionMatrix' )

    if not os.path.isdir(out):
        os.system('mkdir ' + out )
    plt.tight_layout()

    plt.savefig(out + target + '_' + str(classifier) +  '_' + str(param) + '.png', dpi=150)
    plt.close()



def plot_balance_ds(df, dataset):
    fs = 17

    ticks = { 'asteroids' : ['Hazardous' , 'Non Hazardous'],
              'drugs': ["Never", ">10 Years Ago", "Last Decade", "Last Year", "Last Month",
              "Last Week", "Last Day"],
              'breastCancer' :  ['True' , 'False'] ,
              'advertisingBidding' : ['True','False']
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

        fig, ax = plt.subplots(figsize = (11,7))
        width = 0.7
        cum_sum = np.full( len(res[0]), 0)

        for i in range(0,7):
             ax.bar ([''], res[i], width, label=Labels[i], bottom = cum_sum)
             cum_sum += np.array(res[i])


        ax.set_xticklabels(drugs, rotation=35, fontsize=10 )

    else:
        width = 0.7
        data = df[features[dataset]['target']]
        f = np.count_nonzero(df[features[dataset]['target']] == False )
        t = np.count_nonzero(df[features[dataset]['target']] == True )

        fig, ax = plt.subplots(figsize = (11,7))

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



def clean_fast():
    ds = pd.read_csv('input_data/advertisingBidding.shuf.lrn.csv').dropna()
    for rem in ['RowID', 'UserID', 'BidID', 'IP', 'Domain', 'URL', 'Time_Bid', 'AdslotID']:
        del ds[rem]
    labelencoder = LabelEncoder()
    for cl in ['Browser', 'Adslotvisibility', 'Adslotformat']:
        # cleanedTest[cl] = labelencoder.fit_transform(cleanedTest[cl].astype(str))
        ds[cl] = labelencoder.fit_transform(ds[cl].astype(str))

    return ds


""" Main input parameters to choose """

classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB']
classifiers = ['GaussianNB']
classifiers = ['DecisionTree']

dataset = 'drugs'
dataset = 'breastCancer'

validation = 'holdout'
dataset = 'breastCancer'
dataset = 'advertisingBidding'

classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB']
dataset = 'breastCancer'
dataset = 'advertisingBidding'

dataset = 'breastCancer'
dataset = 'asteroids'

#dataset = 'drugs'

def main():

    # Loading the dat frames
    if dataset in ['asteroids', 'drugs']:
        ds = importDataset(dataset)
    else:
        ds = importDataset(dataset, full_chain= True)

        # need to implement the separate testing only
        # xtrain, ytrain, xtest, ytest = importDataset(dataset)


    report_summary = []

    for target in features[dataset]['target']:

        if dataset in ['asteroids', 'drugs', 'advertisingBidding', 'breastCancer']:  # must split the data into train-test
            # Simple Hold Out
            # TO DO Implement: data reshuffling
            if validation == 'holdout':

                a = plot_balance_ds(ds, dataset)

                if dataset == 'drugs':

                    #a = plot_balance_ds(ds, dataset)

                    x,y,x_train,x_test,y_train,y_test = splitDataset(dataset= ds,
                                                                    train_features= features[dataset]['features'],
                                                                    target_features= target )
                else:
                    x, y, x_train, x_test, y_train, y_test = splitDataset(dataset=ds,
                                                                          train_features=features[dataset]['features'],
                                                                          target_features=features[dataset]['target'])

                print('*** Split dataset ***')
            else:
                0 # TO DO implement cross validation

        else:  # data already split
            x_train, x_test, y_train, y_test = xtrain, xtest, ytrain, ytest

            print(0)


        for classifier in classifiers:
            if classifier == 'DecisionTree' : # Run DecisionTreeClassifier
                for param in ['gini'] :  # run the classifier with two different parameter
                #for param in ['gini', 'entropy']:  # run the classifier with two different parameter

                    cf = Classifier(x_train,y_train, classifier=classifier, criterion=param )
                    print("results of " + param + " Index for " + target )
                    y_prediction=predict(x_test,cf,target)
                    confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                    printMatrix(target, confusion_m, classifier, param, dataset)
                    #tree = plot_t(target, dataset, cf)

            if classifier == 'KNeighbors':
                for param in [5, 10 , 50]:
                    cf = Classifier(x_train,y_train, classifier=classifier, n_neighbors=param )
                    print("results of " + str(param) + " Index for " + target )
                    y_prediction=predict(x_test,cf,target)
                    confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                    printMatrix(target, confusion_m, classifier, param, dataset)

            if classifier == 'GaussianNB':
                for param in ['naiveB']:
                    cf = Classifier(x_train,y_train, classifier=classifier)
                    print("results of " + param + " Index for " + target )
                    y_prediction=predict(x_test,cf,target)
                    confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                    printMatrix(target, confusion_m, classifier, param, dataset)


    #dummy = plot_reports(report_summary, classifier, dataset)

if __name__=="__main__":
    main()



