""" Exercise0: dataset descrition
Dataset: Cuff-Less Blood Pressure Estimation
source: https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation
"""

import os,sys
import numpy as np
import pandas as pd
import h5py
import glob
import scipy , scipy.io
import matplotlib.pyplot as plt

files = glob.glob('../data/blood_pressure/Part_*')

print(files)




def get_mat_data(files_list):
    """ Reads the input matlab file and returns numpay arrays.
          input: [] list of files
          output: dictionary of arrays, where each key is a variable, and each values a numpy array """
    
    dataframes = []
    
    for f in files_list:
        
        r = h5py.File(f)
        
        k = [s for s in r.keys() if 'Part' in s  ][0]
        ref = r[k][0][0]
        
        res = np.array(r[ref])
        ppg = [a[0] for a in res ]
        abp = [a[1] for a in res ]
        ecg = [a[2] for a in res ]
        
        dic = {'PPG':ppg  , 'ABP':abp, 'ECG':ecg}
        df = pd.DataFrame.from_dict(dic)
        dataframes.append(df)
        
    """ Creating one unique df out of the available """
    if len(dataframes) >1:
        df = pd.concat(dataframes)
    #f = scipy.io.loadmat(file)
    print(' Succesfully read the file and created the data frame ' )
    return df 
    
    
data = get_mat_data(files)



def analyze_data(data):
    """ Extracts some statistical information from the data """
    
    for c in data.columns:
        values = data[c]
        
        cleaned = [v for v in values if not np.isnan(v)]
        nans = [v for v in values if  np.isnan(v)] 
        mean = np.mean(cleaned)
        median = np.median(cleaned)
        
        print("Valid entries: ", len(cleaned) )
        print("Nans : ", len(nans) )
        print("Mean : ", mean )
        print("Median : ", median )
        
        print(0)
    
#dummy = analyze_data(data)


def plot(data):
    if not os.path.isdir('Plots'):
        os.mkdir('Plots')
        
    """ Main function to create summary plots """
    plot_prop = {'PPG': {'l' : 'Photoplethysmograph [PPG]'      , 'c': 'orange' },
                         'ABP' : {'l' :'Arterial Blood Pressure [ABP]'    , 'c': 'cyan'    },
                         'ECG' : {'l' : 'Raw Electrocardiogram [ECG]'  , 'c':'lime'     } 
                         }
    
    fs = 15
    
    def histo(data):
        """ Plot distributions """
        for c in data.columns:
            plt.hist(data[c] , bins = 30, histtype = 'stepfilled' , color =  plot_prop[c]['c'] , alpha = 0.7 )
            plt.xlabel( plot_prop[c]['l'], fontsize = fs)
            plt.ylabel( 'Counts', fontsize = fs)
            
            plt.grid(ls = ':', color = 'lightgray')
            plt.savefig('Plots/blood_pressure/histo_' + c + '.png' , dpi = 150)
            plt.close()
        
    def timeseries(data):
        for c in data.columns:
            
            plt.plot(data[c][30000:40000] , color =  plot_prop[c]['c'] , alpha = 0.7 )
            plt.ylabel( plot_prop[c]['l'], fontsize = fs)         
            plt.xlabel( 'Data Point',  fontsize = fs)         
            
            if c == 'PPC':
                plt.ylim(0,5)
            elif c == 'ECG':
                plt.ylim(0,2)
            elif c== 'ABP':
                plt.ylim(0,200)
                
            plt.grid(ls = ':', color = 'lightgray')
            plt.savefig('Plots/blood_pressure/series_' + c + '.png' , dpi = 150)        
            plt.close()
    dummy = histo(data)
    dummy = timeseries(data)
    
make_plots = plot(data)
    
print('Done!')