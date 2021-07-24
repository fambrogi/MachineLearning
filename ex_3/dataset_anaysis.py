import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import os,sys

from clean_datasets import dic

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


def make_histos(ds):
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

make_histos('social')
make_histos('income')
make_histos('titanic')