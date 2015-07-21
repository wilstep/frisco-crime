import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from datetime import datetime
from fourier import Fourier

twopi = 2.0 * math.pi
t0 = np.datetime64("2003-01-01")       


## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)

#crime_category = df['Category'] # list of crime types
#group = df.groupby('Category')
#freq = group.size()   # histogram of crime types
#cr_index = freq.index.values  
#Nc = len(cr_index)    # number of crime types
#cr_a_index = pd.DataFrame(data = np.arange(Nc, dtype=np.int), index = cr_index) 
#cr_a_index = cr_a_index[0] # this now holds the index for each crime type

MyF = Fourier(645.0, 20, 52.1775, 12, 1.0, 6, 1.0/7.0, 6, df)
MyF.compute()
MyF.graph()  

## read testing file
#z = zipfile.ZipFile('../test.csv.zip')
#df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)


