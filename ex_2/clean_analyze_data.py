""" Module for cleaning and analyzing the data sets """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
data = { 'data/student-mat.csv': ['G1', 'G2', 'G3'],
         'data/Life_Expectancy_Data.csv': ['boh'],
         'data/Fish.csv': ['weight'],
}


ds_names = ['math', 'fish', 'life', 'wind']
ds_path= ['data/student-mat.csv', 'data/Fish.csv', 'data/Life_Expectancy_Data.csv', 'data/wind_train_data.csv']


def histo(name='', column='' , df = '' ):
    os.system('mkdir Plots/')
    os.system('mkdir Plots/data/')
    os.system('mkdir Plots/data/' + n)

    fs = 12
    plt.hist(df[column], label=column)
    plt.legend()
    plt.grid(ls=':', color='lightgray')
    plt.savefig('Plots/data/' + name + '/' + name + '_histo_' + c.replace('/', '_') + '.png', dpi=150)
    plt.xlabel(column, fontsize=fs)
    plt.close()


def clean(name, df):

    if name=='life': # removing "/" character from column name
        df.rename(columns= {' HIV/AIDS': 'HIV_AIDS'}, inplace=True)
        print(0)

    df = df.dropna() # removing nans from dataframe

    return df


for n,p in zip(ds_names, ds_path):
    df = pd.read_csv(p)
    df = clean(n,df)

    for c in df.columns:
        if 'id' in c or 'time' in c:
            continue
        dummy = histo(name=n, column= c , df = df )



