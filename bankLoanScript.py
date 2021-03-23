import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel("C:/Users/Matteo/Desktop/MachineLearning/Bank_Personal_Loan_Modelling.xlsx",sheet_name='Data')


for c in dataset.columns:
    plt.hist(dataset[c], bins=30, histtype='stepfilled', alpha=0.7)
    plt.xlabel(c, fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.grid(ls=':', color='lightgray')
    plt.savefig("bankPlots/" + c + '.png', dpi=150)
    plt.close()

    # cleaning values
    temp = dataset[c].loc[dataset[c] != 0]
    plt.hist(temp, bins=30, histtype='stepfilled', alpha=0.7)
    plt.xlabel(c, fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.grid(ls=':', color='lightgray')
    plt.savefig('bankPlots/' + c + '_cleaned.png', dpi=150)
    plt.close()