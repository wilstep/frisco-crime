import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import copy
from datetime import datetime
from posd_2 import posd

twopi = 2.0 * math.pi
t0 = np.datetime64("2003-01-01") # 00:00 hours, Wednesday   
swk = 7.0 * 24.0 * 3600.0 # seconds per week

## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
nc = len(df)
print "%d train data crimes in total" % nc
group = df.groupby(['PdDistrict', 'Category'])
freq_pd = group.size()
PDs = df["PdDistrict"].drop_duplicates()
PDs = PDs.tolist() # now have list of PD names
NPDs = len(PDs)
print "There are %d police departments" % NPDs
by_address = df.groupby(['Address', 'Category'])
N_add = df.groupby('Address').size()
freq_add = by_address.size()
# Do Training 
cr_index = freq_pd[PDs[0]].index.values # 0 happens to have all crime categories in it
Ncc = len(cr_index)
mypos = posd(PDs, cr_index, NPDs, Ncc)
mypos.train(freq_pd, freq_add, N_add)
# Training complete

# Read the test data
z = zipfile.ZipFile('../test.csv.zip')
df = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)
nc = len(df)
PDsTest = df.PdDistrict.tolist()
print "%d test data crimes in total" % nc
PDsi = df["PdDistrict"]
# Output predictions
PD_a_index = pd.DataFrame(data = np.arange(NPDs, dtype=np.int), index = PDs) 
PD_a_index = PD_a_index[0] # put PD in, get int out
s1 = np.append(['Id'], [cr_index])
pr = 0
with open('out.csv','wb') as f_handle:
   csvw = csv.writer(f_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
   csvw.writerow(s1)
   for i in range(0,nc):
      s2 = '%d,' % i
      f_handle.write(s2)
      pr = mypos.predict(PDsi[i], df.Address[i])
      csvw.writerow(pr)
      if (i) % 50000 == 0:
         print "i = %d" % i







# first_date = data.date.values[0]









