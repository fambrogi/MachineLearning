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
#import graphviz
from sklearn import tree

""" Define the output directory for Plots (global) """
plot_dir = 'Plots'
if not os.path.isdir(plot_dir):
    os.system('mkdir ' + plot_dir )


def importDataset(ds):
    """ Directory with the datasets paths """
    datasets = {'drugs'     : "input_data/drug_consumption.data.cleaned.csv" ,
                'asteroids' : "input_data/asteroids_cleaned.csv"}

    """ Return a pandas dataframe """
    dataset = pd.read_csv( datasets[ds])
    return dataset



def splitDataset(dataset = '', train_features = [], target_features = [] ):
    """ Split the dataset provided into train and test """

    train_set  = dataset[train_features]
    target_set = dataset[target_features]

    x_train, x_test, y_train, y_test = train_test_split(train_set, target_set, test_size=0.30)
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
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
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
              'drugs': ['0', '1', '2', '3', '4', '5', '6'] }

    classes = ticks[dataset]

    classes = range(0,len(matrix))
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.title(str(classifier) + ' Confusion Matrix - ' + target , y = 1.15, fontsize = fs )

    plt.ylabel('True label', size=fs)
    plt.xlabel('Predicted label', size=fs)

    # Save the fig
    out = plot_dir + '/ConfusionMatrix/' + dataset + '/'

    if not os.path.isdir(out):
        os.system('mkdir ' + out )
    plt.tight_layout()

    plt.savefig(out + target + '_' + str(classifier) +  '_' + str(param) + '.png', dpi=150)
    plt.close()


""" Dictionary containing the training and prediction features """

features = { 'drugs' : { 'train':  ['age', 'gender', 'education', 'ethnicity', 'Nscore',
                                    'Escore', 'Oscore', 'Ascore',
                                    'Cscore', 'Impulsive', 'SS'] ,

                         'target': ['alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine', 'caffeine',
                                    'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy', 'heroin',
                                    'ketamine', 'legal', 'LSD',
                                    'methadone', 'mushrooms', 'nicotine', 'volatileSubstance'] } ,

             'asteroids' : { 'train': ['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)',
                                       'Relative Velocity km per sec', 'Miss Dist.(kilometers)',
                                       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
                                       'Eccentricity', 'Semi Major Axis', 'Inclination', 'Asc Node Longitude',
                                       'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
                                       'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion'] ,

                             'target':  ['Hazardous']

}


             }





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

def plot_tree(feature, dataset, classifier):

    dot_data = tree.export_graphviz(classifier, out_file='Plots/tree_' + feature, filled=True, rounded=True, feature_names=features[dataset]['train'],
                                    class_names= classes  )
    graph = graphviz.Source(dot_data)
    graph.render()

def plot_t(feature, dataset, classifier):
    classes = ['0','1','2','3','4','5', '6']

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(classifier,
                       feature_names= features[dataset]['train'],
                       class_names= classes,
                       filled=True)
    plt.savefig('Plots/tree_' + feature)



classifiers = ['KNeighbors', 'DecisionTree', 'GaussianNB']

classifiers = ['GaussianNB']

classifiers = ['DecisionTree']

dataset = 'drugs'

dataset = 'asteroids'

def main():

    ds = importDataset(dataset)

    report_summary = []

    for target in features[dataset]['target']:

        x,y,x_train,x_test,y_train,y_test = splitDataset(dataset= ds,
                                                         train_features= features[dataset]['train'],
                                                         target_features= target)

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
                    evaluation(y_test, y_prediction)
                    printMatrix(target, confusionMatrix, classifier, param, dataset)

            if classifier == 'GaussianNB':
                for param in ['naiveB']:
                    cf = Classifier(x_train,y_train, classifier=classifier)
                    print("results of " + param + " Index for " + target )
                    y_prediction=predict(x_test,cf,target)
                    confusion_m, accuracy, report = evaluation(y_test, y_prediction )
                    printMatrix(target, confusion_m, classifier, param, dataset)

                    report_summary.append(report)

    #dummy = plot_reports(report_summary, classifier, dataset)

if __name__=="__main__":
    main()



