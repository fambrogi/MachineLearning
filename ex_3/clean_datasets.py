import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import os,sys

def cleanIncome():
	dataset = pd.read_csv("input_data/" + 'adult.data')

	dataset.columns=["age","workclass","fnlwgt","education","education-num","marital-status","occupation"
		,"relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","class"]

	print("printing shape")
	print(dataset.shape)
	print("printing types")
	print(dataset.dtypes)
	# select numeric columns
	datasetNumeric = dataset.select_dtypes(include=[np.number])
	numericCols = datasetNumeric.columns.values
	print("printing numeric columns")
	print(numericCols)
	# select non numeric columns
	datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
	notNumericColums = datasetNotNumeric.columns.values
	print("printing non numeric columns")
	print(notNumericColums)


	for col in notNumericColums:
		dataset[col]=dataset[col].astype("category")
		dataset[col]=dataset[col].cat.codes


	for col in dataset.columns:
		pct_missing = np.mean(dataset[col].isnull())
		print('{} - {}%'.format(col, round(pct_missing * 100)))
	print("missing values")
	print(len(dataset.index))
	lowInfoCols = []
	numRows=dataset.shape[0]
	for col in dataset.columns:
		counts = dataset[col].value_counts(dropna=False)
		top_pct = (counts / numRows).iloc[0]
		if top_pct > 0.95:
			lowInfoCols.append(col)
			print('{0}: {1:.5f}%'.format(col, top_pct * 100))
			print(counts)
			print()
	print("printing low info columns")
	print(lowInfoCols)

	print(dataset.shape)

	dataset.to_csv('input_data/' + 'adult' + '_cleaned.csv' , index = False)

def cleanTitanic():
	dataset = pd.read_csv("input_data/" + 'titanic.csv')
	print("printing shape")
	print(dataset.shape)
	print("printing types")
	print(dataset.dtypes)
	# select numeric columns
	datasetNumeric = dataset.select_dtypes(include=[np.number])
	numericCols = datasetNumeric.columns.values
	print("printing numeric columns")
	print(numericCols)
	# select non numeric columns
	datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
	notNumericColums = datasetNotNumeric.columns.values
	print("printing non numeric columns")
	print(notNumericColums)

	for col in notNumericColums:
		dataset[col] = dataset[col].astype("category")
		dataset[col] = dataset[col].cat.codes

	for col in dataset.columns:
		pct_missing = np.mean(dataset[col].isnull())
		print('{} - {}%'.format(col, round(pct_missing * 100)))
	print("missing values")
	print(len(dataset.index))
	lowInfoCols = []
	numRows = dataset.shape[0]
	for col in dataset.columns:
		counts = dataset[col].value_counts(dropna=False)
		top_pct = (counts / numRows).iloc[0]
		if top_pct > 0.95:
			lowInfoCols.append(col)
			print('{0}: {1:.5f}%'.format(col, top_pct * 100))
			print(counts)
			print()
	print("printing low info columns")
	print(lowInfoCols)
	#the name and id of the buyer are not useful i proceed with the deletion of the two columns
	del(dataset['PassengerId'])
	del (dataset['Name'])
	dataset = dataset.dropna()
	print(dataset.shape)

	dataset.to_csv('input_data/' + 'titanic' + '_cleaned.csv', index=False)

def cleanAds():
	dataset = pd.read_csv("input_data/" + 'Social_Network_Ads.csv')
	print("printing shape")
	print(dataset.shape)
	print("printing types")
	print(dataset.dtypes)
	# select numeric columns
	datasetNumeric = dataset.select_dtypes(include=[np.number])
	numericCols = datasetNumeric.columns.values
	print("printing numeric columns")
	print(numericCols)
	# select non numeric columns
	datasetNotNumeric = dataset.select_dtypes(exclude=[np.number])
	notNumericColums = datasetNotNumeric.columns.values
	print("printing non numeric columns")
	print(notNumericColums)

	for col in notNumericColums:
		dataset[col] = dataset[col].astype("category")
		dataset[col] = dataset[col].cat.codes

	for col in dataset.columns:
		pct_missing = np.mean(dataset[col].isnull())
		print('{} - {}%'.format(col, round(pct_missing * 100)))
	print("missing values")
	print(len(dataset.index))
	lowInfoCols = []
	numRows = dataset.shape[0]
	for col in dataset.columns:
		counts = dataset[col].value_counts(dropna=False)
		top_pct = (counts / numRows).iloc[0]
		if top_pct > 0.95:
			lowInfoCols.append(col)
			print('{0}: {1:.5f}%'.format(col, top_pct * 100))
			print(counts)
			print()
	print("printing low info columns")
	print(lowInfoCols)
	# the user Id gives us no info
	del (dataset['User ID'])

	#dataset = dataset.dropna()
	print(dataset.shape)

	dataset.to_csv('input_data/' + 'ads' + '_cleaned.csv', index=False)



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

		if not os.path.isdir("Plots/preprocessing_plots/"  ):
			os.mkdir("Plots/preprocessing_plots/"  )

		if not os.path.isdir("Plots/preprocessing_plots/" + folderName ):
			os.mkdir("Plots/preprocessing_plots/" + folderName )


		plt.savefig("Plots/preprocessing_plots/" + folderName + '/' + col + '.png', dpi=150)
		plt.close()

	# due the fact that numerical attributes are numerical but they corresponds to intervals I print histograms also for them in order to
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





def main():
	print("cleaning income")
	cleanIncome()
	print("cleaning titanic")
	cleanTitanic()
	print("cleaning ads")
	cleanAds()


if __name__ == "__main__":
	main()
