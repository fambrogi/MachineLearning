import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def clean_drug():
	dataset = pd.read_csv("input_data/" + 'drug_consumption.data')
	dataset.columns = ['ID', 'age', 'gender', 'education', 'country', 'ethnicity', 'Nscore', 'Escore', 'Oscore',
					   'Ascore',
					   'Cscore', 'Impulsive', 'SS', 'alcohol', 'amphetamines', 'amylNitrite', 'benzodiazepine',
					   'caffeine',
					   'cannabis', 'chocolate', 'cocaine', 'crack', 'ecstasy',
					   'heroin', 'ketamine', 'legal', 'LSD', 'methadone', 'mushrooms', 'nicotine',
					   'semeron', 'volatileSubstance']
	out = open('input_data/' + 'drugs_preparation_summary.txt', 'w')
	to_write = []

	# shape and data types of the data
	to_write.append("shape of the dataset\n")
	to_write.append(dataset.shape)


	to_write.append("\ntypes of the objects\n")
	to_write.append(dataset.dtypes)

	# select numeric columns
	datasetNumeric = dataset.select_dtypes(include=[np.number])
	numericCols = datasetNumeric.columns.values
	to_write.append("\nList of the numeric columns\n")
	to_write.append(numericCols)

	# select non numeric columns
	datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
	notNumericColums = datasetNotNumeric.columns.values
	to_write.append("\nlist of the non numeric Columns\n")
	to_write.append(notNumericColums)

	# I change the label from CL0 to 0 CL1 into 1 and so on
	for c in notNumericColums:
		dataset[c] = dataset[c].map({'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6}).astype(int)

	# missing values percentage
	to_write.append("\npercentage of missing values\n")
	for col in dataset.columns:
		pct_missing = np.mean(dataset[col].isnull())
		to_write.append('{} - {}%'.format(col, round(pct_missing * 100)))
		to_write.append("\n")

	to_write.append('\nlow information columns 90% of observation are equal\n')
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

	to_write.append(lowInfoCols)

	# we have no duplicate observations
	# we know that column 'id' is unique, but what if we drop it?
	df_dedupped = dataset.drop('ID', axis=1).drop_duplicates()
	# there were duplicate rows
	print(dataset.shape)
	print(df_dedupped.shape)

	del dataset['semeron']
	del dataset['country']
	del dataset['ID']

	dataset.to_csv('input_data/' + 'drug_consumption' + '_cleaned.csv')
	for s in to_write:
		out.write(str(s))
	out.close()


def clean_asteroids():
	dataframe = pd.read_csv("input_data/" + 'asteroids.csv')
	out = open('input_data/' + 'asteroids_preparation_summary.txt', 'w')
	to_write = []

	col_to_drop = ['Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
				   'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Neo Reference ID', 'Name', 'Close Approach Date',
				   'Epoch Date Close Approach', 'Relative Velocity km per hr', 'Miles per hour',
				   'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)',
				   'Miss Dist.(miles)', 'Equinox', 'Orbit Determination Date', 'Orbiting Body', 'Epoch Osculation',
				   'Orbit ID', 'Orbit Uncertainity']

	# will be kept:
	# ['Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)',
	#        'Relative Velocity km per sec', 'Miss Dist.(kilometers)',
	#        'Orbiting Body', 'Orbit ID', 'Orbit Determination Date',
	#        'Orbit Uncertainity', 'Minimum Orbit Intersection',
	#        'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity',
	#        'Semi Major Axis', 'Inclination', 'Asc Node Longitude',
	#        'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
	#        'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion',
	#        'Hazardous']

	for c in col_to_drop:
		del dataframe[c]

	""" Dropping nans, checking how many rows are removed """
	LenBeforeNanDrop = len(dataframe)
	dataset = dataframe.dropna()
	LenAfterNanDrop = len(dataframe)

	to_write.append('LenBeforeNanDrop' + str(LenBeforeNanDrop))
	to_write.append('LenAfterNanDrop' + str(LenAfterNanDrop))

	""" Counting how many hazardous/non haz inside """
	labels, counts = np.unique(dataframe['Hazardous'], return_counts=True)
	to_write.append(str(labels[0]) + '_' + str(counts[0]))
	to_write.append(str(labels[1]) + '_' + str(counts[1]))

	for s in to_write:
		out.write(s + '\n')
	out.close()
	""" Save cleaned dataframe """
	dataframe.to_csv('input_data/' + 'asteroids_cleaned.csv', index=False)
	return 0


def printBasicInfo(dataset):
	# shape and data types of the data
	toPrint=[]
	toPrint.append(dataset.shape)
	toPrint.append(dataset.dtypes)
	return toPrint


def devideNumericCols(dataset):
	# select numeric columns
	datasetNumeric = dataset.select_dtypes(include=[np.number])
	numericCols = datasetNumeric.columns.values
	# select non numeric columns
	datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
	notNumericColums = datasetNotNumeric.columns.values
	return numericCols, notNumericColums


def findMissingValues(dataset):
	# missing values percentage
	toPrint=[]
	toPrint.append("\nprinting missing values\n")
	for col in dataset.columns:
		pct_missing = np.mean(dataset[col].isnull())
		toPrint.append('{} - {}%'.format(col, round(pct_missing * 100)))
	return toPrint


def plotOutliersForContinuosData(dataset, notNumericCols, numericCols, folderName):
	# outliers for categorical attributes

	for col in notNumericCols:
		dataset[col].value_counts().plot.bar()
		plt.title(col)
		plt.plot()
		plt.savefig("Plots/preprocessing_plots/" + folderName + '/' + col + '.png', dpi=150)
		plt.close()

	# due the fact that numerical attributes are numerical but they corresponts at intervals I print histograms also for them in order to
	# spot outliers
	for col in numericCols:
		dataset.boxplot(column=col)
		plt.title(col)
		plt.plot()
		plt.savefig("Plots/preprocessing_plots/" + folderName + '/' + col + '.png', dpi=150)
		plt.close()


def findLowInfoCols(dataset):
	toPrint=[]
	toPrint.append('\n'+'low information columns'+'\n')
	numRows = len(dataset.index)
	lowInfoCols = []
	for col in dataset.columns:
		counts = dataset[col].value_counts(dropna=False)
		top_pct = (counts / numRows).iloc[0]
		if top_pct > 0.80:
			lowInfoCols.append(col)
			toPrint.append(col)
			toPrint.append('{0}: {1:.5f}%'.format(col, top_pct * 100))
			toPrint.append(counts)
			toPrint.append("\n")
	return toPrint,lowInfoCols


def findDuplicatedValues(dataset, indexCol):
	toPrint=[]
	df_dedupped = dataset.drop(indexCol, axis=1).drop_duplicates()
	# there were duplicate rows
	toPrint.append('\n'+"finding duplicates"+'\n')
	toPrint.append(dataset.shape)
	toPrint.append(df_dedupped.shape)
	return toPrint


def cleanBreastCancer():
	testSet = pd.read_csv("input_data/breast-cancer-diagnostic.shuf.tes.csv")
	trainSet = pd.read_csv("input_data/breast-cancer-diagnostic.shuf.lrn.csv")
	solutionSet = pd.read_csv("input_data/breast-cancer-diagnostic.shuf.sol.ex.csv")
	out = open('input_data/' + 'breastCancer_preparation_summary.txt', 'w')

	generalFrame = pd.concat([trainSet, testSet])
	toWrite=[]

	toWrite.append(printBasicInfo(generalFrame))

	findMissingValues(generalFrame)
	findDuplicatedValues(generalFrame, 'ID')
	toPrint,lowInfoCols = findLowInfoCols(generalFrame)
	toWrite.append(toPrint)
	# In this dataset the numerical values are not organized in ranges but we can have values in all the domain
	# we don't have intervals instead of instograms is better use boxplots
	generalFrame = pd.concat([trainSet, testSet])
	numericC, nNumericC = devideNumericCols(generalFrame)
	plotOutliersForContinuosData(generalFrame, nNumericC, numericC, 'breastCancer')

	cleanedTrain = trainSet
	cleanedTest = testSet
	cleanedSolution = solutionSet
	del cleanedTrain['ID']
	del cleanedTest['ID']
	del cleanedSolution['ID']
	cleanedTrain.to_csv('input_data/cleanedTrain.csv')
	cleanedTest.to_csv('input_data/cleanedTest.csv')
	cleanedSolution.to_csv('input_data/cleanedSolution.csv')
	for s in toPrint:
		out.write(str(s))
	out.close()


def plotOutliersNotContinuous(dataset, notNumericCols, numericCols, folderName):
	# outliers for categorical attributes

	for col in notNumericCols:
		print(col)
		dataset[col].value_counts().plot.bar()
		plt.title(col)
		plt.plot()
		plt.savefig("Plots/preprocessing_plots/" + folderName + '/' + col + '.png', dpi=150)
		plt.close()

	# due the fact that numerical attributes are numerical but they corresponts at intervals I print histograms also for them in order to
	# spot outliers
	for col in numericCols:
		dataset[col].value_counts().plot.bar()
		plt.title(col)
		plt.plot()
		plt.savefig("Plots/preprocessing_plots/" + folderName + '/' + col + '.png', dpi=150)
		plt.close()


def cleanOnlineAdvertisement():
	testSet = pd.read_csv("input_data/advertisingBidding.shuf.tes.csv")
	trainSet = pd.read_csv("input_data/advertisingBidding.shuf.lrn.csv")
	solutionSet = pd.read_csv("input_data/advertisingBidding.shuf.sol.ex.csv")
	toWrite=[]
	out = open('input_data/' + 'onlineAdvertisement_preparation_summary.txt', 'w')

	generalFrame = pd.concat([trainSet, testSet])

	toWrite.append(printBasicInfo(generalFrame))

	findMissingValues(generalFrame)
	findDuplicatedValues(generalFrame, 'RowID')
	toPrint,lowInfoCols = findLowInfoCols(generalFrame)
	toWrite.append(toPrint)

	# in the Url the 4% of the observation is null I delete those rows they are 1000 on 25000
	cleanedTrain = generalFrame[trainSet['URL'].notnull()]
	rowsToDelete = generalFrame[trainSet['URL'].isnull()].loc[:, 'RowID']
	cleanedTest = generalFrame[testSet['URL'].notnull()]
	# I have to remove the rows in the solution set that I have deleted from the testSet
	cleanedSolution = solutionSet[~solutionSet.RowID.isin(rowsToDelete.tolist())]
	# The browser column creates issues due its values I want to solve it encoding the column
	labelencoder = LabelEncoder()

	""" Converting each class to string, since nans are considered as float, hence it cerates a conflict wth object types """
	for cl in ['Browser', 'Adslotvisibility', 'Adslotformat']:
		cleanedTest[cl] = labelencoder.fit_transform(cleanedTest[cl].astype(str))
		cleanedTrain[cl] = labelencoder.fit_transform(cleanedTrain[cl].astype(str))

	# I remove all this columns because we can assume that they depend by the single observation thay only create noise
	# and not relevant data

	for rem in ['RowID', 'UserID', 'BidID', 'IP', 'Domain', 'URL', 'Time_Bid', 'AdslotID']:
		del cleanedTest[rem]
		del cleanedTrain[rem]
	del cleanedSolution['RowID']

	generalFrame = pd.concat([cleanedTrain, cleanedTest])
	numericC, nNumericC = devideNumericCols(generalFrame)
	plotOutliersForContinuosData(generalFrame, nNumericC, numericC, 'onlineAdvertisement')

	cleanedTrain = cleanedTrain.dropna()
	cleanedTest = cleanedTest.dropna()
	cleanedSolution = cleanedSolution.dropna()

	cleanedTrain.to_csv('input_data/cleanedTrain.csv')
	cleanedTest.to_csv('input_data/cleanedTest.csv')
	cleanedSolution.to_csv('input_data/cleanedSolution.csv')
	for s in toWrite:
		out.write(str(s))
	out.close()

def main():
	clean_drug()
	cleanBreastCancer()
	cleanOnlineAdvertisement()
	clean_asteroids()


if __name__ == "__main__":
	main()
