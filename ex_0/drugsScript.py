import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys

dataset = pd.read_csv("data/drug_consumption.data")
dataset.columns=['ID', 'age', 'gender', 'education' ,'country', 'ethnicity','Nscore', 'Escore', 'Oscore', 'Ascore',
                 'Cscore', 'Impulsive','SS', 'alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine' ,'caffeine',
                 'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy',
                 'heroin', 'ketamine','legal','LSD', 'methadone','mushrooms','nicotine',
                 'semeron','volatileSubstance']

def bar_drugs():
    if not os.path.isdir('Plots/drugsPlots/'):
        os.system('mkdir Plots/drugsPlots/')

    fs = 12

    labels = ["Never", ">10 Years Ago", "Last Decade", "Last Year", "Last Month",
              "Last Week", "Last Day"]

    for c in ['alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine' ,'caffeine',
                 'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy',
                 'heroin', 'ketamine','legal','LSD', 'methadone','mushrooms','nicotine',
                 'semeron','volatileSubstance']:

        fig= plt.figure()

        counts = dataset[c].value_counts().to_dict()
        values = []

        for v in range(0,7):
            cl = 'CL' + str(v)
            if cl in counts.keys():
                values.append(counts[cl])
            else:
                values.append(0)
        plt.bar(range(0,7), values, color = 'red', alpha=0.6)
        plt.title(c + ' Consumption Frequency' , y=1.02 , fontsize = fs )
        plt.ylabel('Counts', fontsize = fs )
        plt.xticks(range(0,7), labels, rotation = 25, fontsize = 8)
        plt.grid(ls=':', color='lightgray')

        plt.tight_layout()
        plt.savefig("Plots/drugsPlots/" + c + '_freq.png', dpi=150, )


        plt.close()


def bar_personal():
    if not os.path.isdir('Plots/drugsPlots/'):
        os.system('mkdir Plots/drugsPlots/')

    fs = 12
    age = ['18-24', '25-34','35-44','45-54','55-64','65+']
    ed = ['Left before 16','Left at 16','Left at 17','Left at 18',
          'Some college', 'Prof. Certificate',
          'Uni. Degree','Master Degree','Doctoral Degree']
    co = ['Australia','Canada','New Zealand',
          'Other', 'Rep. Ireland','UK','USA']

    for c,l in zip( ['age', 'education', 'country'] , [age,ed,co] ):

        fig= plt.figure()

        counts = dataset[c].value_counts().to_dict()
        values = []

        keys = counts.keys()
        keys.sort()
        for v in keys:
            values.append(counts[v])

        Len = len(values)

        plt.bar(range(0, Len ), values, color = 'orange', alpha=0.6)
        plt.title(c + ' Consumption Frequency' , y=1.02 , fontsize = fs-1 )
        plt.ylabel('Counts', fontsize = fs )
        plt.xticks(range(Len), l, rotation = 25, fontsize = 8)
        plt.grid(ls=':', color='lightgray')

        plt.tight_layout()
        plt.savefig("Plots/drugsPlots/" + c + '_freq.png', dpi=150, )


        plt.close()




def histo_scores():

    fs = 15
    for c in ['Nscore', 'Escore', 'Oscore', 'Ascore',
                 'Cscore', 'Impulsive','SS']:

        plt.hist(dataset[c], bins=20, histtype='stepfilled', alpha=0.7, color = 'lime')
        plt.xlabel(c, fontsize=fs)
        plt.ylabel('Counts', fontsize=fs)
        plt.grid(ls=':', color='lightgray')
        plt.tight_layout()
        plt.savefig("Plots/drugsPlots/" + c + '.png', dpi=150)
        plt.close()



#dummy = bar_drugs()
#dummy = histo_scores()
dummy = bar_personal()


"""
    plt.hist(dataset[c], bins=30, histtype='stepfilled', alpha=0.7)
    plt.xlabel(c, fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.grid(ls=':', color='lightgray')
    plt.savefig("drugsPlots/" + c + '.png', dpi=150)
    plt.close()
"""


"""
for c in dataset.columns:
    plt.hist(dataset[c], bins=30, histtype='stepfilled', alpha=0.7)
    plt.xlabel(c, fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.grid(ls=':', color='lightgray')
    plt.savefig("drugsPlots/" + c + '.png', dpi=150)
    plt.close()
"""
