""" Module for cleaning and analyzing the data sets """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

""" Storing the information of all datasets.
    Column names as they appear after data cleaning. """

data = {'math': {'path': 'data/student-mat.csv',
                 'features': [],
                 'drop': [],
                 'targets' : ['G1', 'G2', 'G3']},

        'life': { 'path': 'data/Life_Expectancy_Data.csv',
                  'features': ['AdultMortality',
                               'infantdeaths', 'Alcohol', 'percentageexpenditure', 'HepatitisB',
                               'Measles', 'BMI', 'under-fivedeaths', 'Polio', 'Totalexpenditure',
                               'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness1-19years',
                               'thinness5-9years', 'Incomecompositionofresources', 'Schooling' ],
                  'drop': ['Country', 'Year', 'Status'],
                  'targets': ['Lifeexpectancy']},

        'wind': {'path': 'data/wind_train_data.csv',
                  'features': ['wind_speed(m/s)',
                               'atmospheric_temperature(°C)', 'shaft_temperature(°C)',
                               'blades_angle(°)', 'gearbox_temperature(°C)', 'engine_temperature(°C)',
                               'motor_torque(N-m)', 'generator_temperature(°C)',
                               'atmospheric_pressure(Pascal)', 'area_temperature(°C)',
                               'windmill_body_temperature(°C)', 'wind_direction(°)', 'resistance(ohm)',
                               'rotor_torque(N-m)', 'blade_length(m)',
                               'blade_breadth(m)', 'windmill_height(m)' ],

                  'drop': ['turbine_status', 'cloud_level', 'tracking_id', 'datetime'],
                  'targets': ['windmill_generated_power(kW_h)']},

        }

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


# perform cleaning on the "math" data set
# One-Hot encoding of multi-categorical-columns
# Replacing yes/no with 1/0
def mathCleaning(df):
    # replace "yes" with 1 and "no" with 0
    df = df.replace(["yes", "no"], [1, 0])

    # replace sex "F" with 1 and "M" with 0
    df["sex"] = df["sex"].replace(["F", "M"], [1, 0])

    # replace Pstatus "A" with 1 and "T" with 0
    df["Pstatus"] = df["Pstatus"].replace(["A", "T"], [1, 0])

    # replace famsize "GT3" with 1 and "LT3" with 0
    df["famsize"] = df["famsize"].replace(["GT3", "LT3"], [1, 0])

    # replace address "U" with 1 and "R" with 0
    df["address"] = df["address"].replace(["U", "R"], [1, 0])

    # replace school "GP" with 1 and "MS" with 0
    df["school"] = df["school"].replace(["GP", "MS"], [1, 0])

    # perform one-hot-encoding for the rest
    df = pd.get_dummies(df)

    return df


""" Main modules to clean the df before training """

def load_clean_data(name):
    """ Main utility to clean and prepare the data """

    df = pd.read_csv(data[name]['path'])

    # Dropping empty space in column names
    for c in df.columns: #
        if c != c.replace(' ','_'):
            df.rename(columns={c: c.replace(' ','')}, inplace=True)
        if '/' in c:
            df.rename(columns={c: c.replace('/','_')}, inplace=True)

    # Clean values from nans
    df = df.dropna()

    # for the "math" data set apply special changes
    if name == "math":
        df = mathCleaning(df)

    print(df.head())

    # Shuffling data
    df = df.sample(frac=1)

    # Print general information on the dataset
    df.describe()

    # Dropping not used columns
    if data[name]['drop']:
        df = df.drop(columns=data[name]['drop'])
    return df




#ds = 'life'
#a = load_clean_data(ds)


"""
# to fix, later
for n,p in zip(ds_names, ds_path):
    df = pd.read_csv(p)
    df = clean(n,df)

    for c in df.columns:
        if 'id' in c or 'time' in c:
            continue
        dummy = histo(name=n, column= c , df = df )
"""
