import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import os,sys


def make_histos_2(ds, df, what = ''):
    """ Creates histograms and correlation matrices """

    if not os.path.isdir('plots'):
        os.system('mkdir plots')
    fs = 12

    red = { 'income' : ["age", "workclass",  "education",
                        "capital-gain", "capital-loss", "class"] ,

            'titanic' : ['Age', 'Ticket', 'Fare', 'Sex', 'Pclass', 'Survived'] }


    print("Analysing and plotting : ", ds )


    if not os.path.isdir('plots'):
        os.system('mkdir plots')


    # print 2 by 2 correlations and histograms
    sns.set_theme(style="ticks")
    sns.pairplot(df, corner=False, diag_kind = "hist")

    plt.tight_layout()
    plt.savefig('plots/' + ds + '_pairplot_' + what + '.png', dpi = 200)
    plt.close()

    # correlation matrix
    # Compute the correlation matrix, using only a subset of columns for some data sets
    corr = df.corr()
    a = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    sns.heatmap(corr,  vmin = -1, vmax = 1, linewidths=.5, cbar_kws={"shrink": .5}, cmap = a)
    plt.title("Correlations for the data set " + ds + ' ' + what , fontsize = fs )
    plt.tight_layout()
    plt.savefig('plots/' + ds + '_correlations_' + what + '.png', dpi = 200)
    plt.close()
    
    """
    if ds in red.keys():
        df_r = df[red[ds]]

        corr = df_r.corr()
        a = sns.diverging_palette(145, 300, s=60, as_cmap=True)
        sns.heatmap(corr, vmin=-1, vmax=1, linewidths=.5, cbar_kws={"shrink": .5}, cmap=a)
        plt.title("Correlations for the data set " + ds + ' ' + what, fontsize=fs)
        plt.tight_layout()
        plt.savefig('plots/' + ds + '_correlations_reduced_' + what + '.png', dpi=200)
        plt.close()

        sns.pairplot(df_r )
        plt.tight_layout()
        plt.savefig('plots/' + ds + '_pairplot_reduced_' + what + '.png', dpi = 200)
        plt.close()
    """
    return 0





def printConfusionMatrix(matrix, ds, title='', what='') :
    """ Plot the confusion matrices resulting from the model training and evaluation """
    if not os.path.isdir('ConfusionMatrixes'):
        os.system('mkdir ConfusionMatrixes')
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

    plt.title( what + ", Dataset " + ds  )

    plt.ylabel('True label', size=fs)
    plt.xlabel('Predicted label', size=fs)
    plt.tight_layout()
    plt.savefig('ConfusionMatrixes/' + ds + '_' + what + '_' + title + '.png', dpi=150)
    plt.close()




def plot_results(res, ds ):
    """ Plot the scores after the training """
    if not os.path.isdir('plots/results'):
        os.system(' mkdir plots/results' )

    fs = 12
    labels = ['Original', 'Gaussian Copula' , 'ctGAN', 'Copula GAN']

    for r,c in zip(['precision', 'accuracy', 'recall'], ['lime', 'orange', 'blue']):
         plt.scatter([1,2,3,4] , res[ds][r], label = r, color = c )
         plt.title (ds + ' Dataset ', fontsize = fs, y = 1.02)
         plt.ylabel(r, fontsize = fs)
         plt.grid(ls = ':' , color = 'lightgray')
         plt.xticks([1,2,3,4] , labels)
         plt.savefig('plots/results/' + ds + '_' + r + '.png' , dpi = 200 )
         plt.close()

    print("Done plotting results *** ")

    
def plot_histo_comparison_ds(ds, columns=''):
    """ Plot a comparison of the histograms of the original and synthetic data sets """
    if not os.path.isdir('plots/histo_check'):
        os.system('mkdir plots/histo_check')


    fs = 12
    original_df = pd.read_csv('splittedDatasets/x_train_' + ds + '.csv')
    copula_df = pd.read_csv('generatedData/x_train_' + ds + '_gaussian_copula.csv')
    GAN_df = pd.read_csv('generatedData/x_train_' + ds + '_ctGAN.csv')
    GAN_copula_df = pd.read_csv('generatedData/x_train_' + ds + '_copulaGAN.csv')

    data = []

    labels = ['Original' , 'Copula', 'GAN', 'Copula GAN']

    colors = ['gold', 'red', 'cyan', 'blue']

    for c in columns:
        data = []
        for df in [original_df , copula_df, GAN_df, GAN_copula_df]:
            h = df[c]
            data.append(h)
        plt.title('Data comparison for ' + ds + ' Dataset' )
        plt.hist(data, histtype='stepfilled' , label= labels, color = colors, bins = 20, alpha = 0.7 )
        plt.xlabel(c , fontsize = fs)
        plt.legend()
        plt.savefig('plots/histo_check/' + ds + '_step_' + c + '.png', dpi = 200)
        plt.close()
    for c in columns:
        data = []
        for df in [original_df , copula_df, GAN_df, GAN_copula_df]:
            h = df[c]
            data.append(h)
        plt.title('Data comparison for ' + ds + ' feature: ' + c )
        plt.hist(data, label= labels, bins = 20 , color = colors)
        plt.legend()
        plt.savefig('plots/histo_check/' + ds + '_' + c + '.png', dpi = 200)
        plt.close()



    return
