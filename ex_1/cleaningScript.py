import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("data/drug_consumption.data")
dataset.columns=['ID', 'age', 'gender', 'education' ,'country', 'ethnicity','Nscore', 'Escore', 'Oscore', 'Ascore',
                 'Cscore', 'Impulsive','SS', 'alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine' ,'caffeine',
                 'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy',
                 'heroin', 'ketamine','legal','LSD', 'methadone','mushrooms','nicotine',
                 'semeron','volatileSubstance']

# shape and data types of the data
print(dataset.shape)
print(dataset.dtypes)

# select numeric columns
datasetNumeric = dataset.select_dtypes(include=[np.number])
numericCols = datasetNumeric.columns.values
print(numericCols)

# select non numeric columns
datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
notNumericColums = datasetNotNumeric.columns.values
print(notNumericColums)

#I change the label from CL0 to 0 CL1 into 1 and so on
for c in notNumericColums:
    dataset[c]=dataset[c].map({'CL0': 0 ,'CL1': 1,'CL2': 2,'CL3': 3 ,'CL4': 4,'CL5': 5,'CL6': 6}).astype(int)

#missing values percentage
for col in dataset.columns:
    pct_missing = np.mean(dataset[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

#outliers search for numeric attributes

#for col in numericCols:
#    dataset.boxplot(column=col)
#    plt.savefig("outliersPlots/" + col+'Boxplot.png', dpi=150)
#    plt.close()

#for col in numericCols:
#    print(dataset[col].describe())

#outliers for categorical attributes
for col in notNumericColums:
    dataset[col].value_counts().plot.bar()
    plt.title(col)
    plt.savefig("outliersPlots/" + col+'Histo.png', dpi=150 )
    plt.close()

#due the fact that numerical attributes are numerical but they corresponts at intervals I print histograms also for them in order to
#spot outliers
for col in numericCols:
    dataset[col].value_counts().plot.bar()
    plt.title(col)
    plt.savefig("outliersPlots/" + col+'Histo.png', dpi=150 )
    plt.close()


print('low information columns')
numRows = len(dataset.index)
lowInfoCols = []

for col in dataset.columns:
    counts = dataset[col].value_counts(dropna=False)
    top_pct = (counts / numRows).iloc[0]

    if top_pct > 0.90:
        lowInfoCols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct * 100))
        print(counts)
        print()

#we have no duplicate observations
# we know that column 'id' is unique, but what if we drop it?
df_dedupped = dataset.drop('ID', axis=1).drop_duplicates()
# there were duplicate rows
print(dataset.shape)
print(df_dedupped.shape)

del dataset['semeron']
del dataset['country']
del dataset['ID']

dataset.to_csv('data/cleanedDataset.csv')



