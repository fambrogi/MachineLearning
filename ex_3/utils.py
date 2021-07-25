import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import os,sys

#from clean_datasets import dic

"""
dic = {"income": {"path": "input_data/adult.data",
				  "features": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
							  "occupation",
							  "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
							  "native-country", "class"],
				  "remove": []},

	   "titanic": {"path": "input_data/titanic.csv",
				   "features": ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
				   "remove": ['PassengerId', 'Name']},

	   "social": {"path": "input_data/Social_Network_Ads.csv",
				  "features": ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'],
				  "remove": ["User ID"] },

	   }
"""


def make_histos(ds, df):
    fs = 12

    print("Analysing and plotting : ", ds )

    if not os.path.isdir('plots'):
        os.system('mkdir plots')
    df = pd.read_csv(dic[ds]['path'])
    for c in dic[ds]['remove']:
        del df[c]

    if ds == "income":
        df.columns = dic[ds]['features']

    print("Columns: " , df.columns )

    # print 2 by 2 correlations and histograms
    sns.set_theme(style="ticks")
    sns.pairplot(df, corner=True )
    plt.savefig('plots/' + ds + '_pairplot.png', dpi = 200)
    plt.close()

    # correlation matrix
    # Compute the correlation matrix
    corr = df.corr()
    a = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    sns.heatmap(corr,  vmin = -1, vmax = 1, linewidths=.5, cbar_kws={"shrink": .5}, cmap = a)
    plt.title("Correlations for the data set " + ds , fontsize = fs )
    plt.savefig('plots/' + ds + '_correlations.png', dpi = 200)
    plt.close()


    return 0


def make_histos_2(ds, df):
    fs = 12

    print("Analysing and plotting : ", ds )

    if not os.path.isdir('plots'):
        os.system('mkdir plots')

    print("Columns: " , df.columns )

    # print 2 by 2 correlations and histograms
    sns.set_theme(style="ticks")
    sns.pairplot(df, corner=True )
    plt.savefig('plots/' + ds + '_pairplot.png', dpi = 200)
    plt.close()

    # correlation matrix
    # Compute the correlation matrix
    corr = df.corr()
    a = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    sns.heatmap(corr,  vmin = -1, vmax = 1, linewidths=.5, cbar_kws={"shrink": .5}, cmap = a)
    plt.title("Correlations for the data set " + ds , fontsize = fs )
    plt.savefig('plots/' + ds + '_correlations.png', dpi = 200)
    plt.close()


    return 0





def printConfusionMatrix(matrix, ds, title='', what='') :

    fs = 12
    plt.figure(figsize=(6, 5))
    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    # plot colorbar to the right
    plt.colorbar()
    fmt = 'd'
    fmt = 'f.2'
    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(matrix[i, j]), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    # all the same in this case
    ticks = {'income'  : ['0', '1'],
             'titanic': ['0', '1'],
             'social' : ['0', '1']}

    #classes = ticks[dataset]

    classes = range(0, len(matrix))
    tick_marks = np.arange(len(classes))

    try:
        plt.xticks(tick_marks, ticks[ds], rotation=20, fontsize=8)
        plt.yticks(tick_marks, ticks[ds], rotation=20, va="center", fontsize=8)
    except:
        pass
    titles = {'income': 'Class',
              'titanic': 'Survive',
              'social': 'Buy'}

    plt.title( titles[ds] + ' for ' + what)

    plt.ylabel('True label', size=fs)
    plt.xlabel('Predicted label', size=fs)
    plt.tight_layout()
    plt.savefig('ConfusionMatrixes/' + ds + '_' + what + '_' + title + '.png', dpi=150)
    plt.close()


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

