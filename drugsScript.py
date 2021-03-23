import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/Matteo/Desktop/MachineLearning/drug_consumption.data")
dataset.columns=['ID', 'age', 'gender', 'education' ,'country', 'ethnicity','Nscore', 'Escore', 'Oscore', 'Ascore',
                 'Cscore', 'Impulsive','SS', 'alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine' ,'caffeine',
                 'cannabis', 'chocolateConsumption', 'cocaine', 'crack', 'ecstasy',
                 'heroin', 'ketamine','legal','LSD', 'methadone','mushrooms','nicotine',
                 'semeron','volatileSubstanceAbuse']


for c in dataset.columns:
    plt.hist(dataset[c], bins=30, histtype='stepfilled', alpha=0.7)
    plt.xlabel(c, fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.grid(ls=':', color='lightgray')
    plt.savefig("drugsPlots/" + c + '.png', dpi=150)
    plt.close()