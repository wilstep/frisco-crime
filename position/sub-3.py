import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import copy
from datetime import datetime
from posd_3 import posd


## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df1 = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
# test data
z = zipfile.ZipFile('../test.csv.zip')
df2 = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)
# have loaded all the data
df = df1[df1.Y < 38.0]
X = df.X
Y = df.Y
xmax = X.max()
xmin = X.min()
ymax = Y.max()
ymin = Y.min()
#Check out the test data
df = df2[df2.Y < 38.0]
X = df.X
Y = df.Y
xmax = max(xmax, X.max()) + 0.0000001
xmin = min(xmin, X.min()) - 0.0000001
ymax = max(ymax, Y.max()) + 0.0000001
ymin = min(ymin, Y.min()) - 0.0000001
print "Global extremes"
print "Xmax = %f, Ymax = %f" % (xmax, ymax)
print "Xmin = %f, Ymin = %f" % (xmin, ymin)
# max's and mins found, will be used for grid
nc = len(df1)
print "%d train data crimes in total" % nc
group = df1.groupby(['PdDistrict', 'Category'])
freq_pd = group.size()
PDs = df1["PdDistrict"].drop_duplicates()
PDs = PDs.tolist() # now have list of PD names
NPDs = len(PDs)
print "There are %d police departments" % NPDs
by_address = df1.groupby(['Address', 'Category'])
N_add = df1.groupby('Address').size() # number of crimes per address
freq_add = by_address.size()
# Do Training 
cr_index = freq_pd[PDs[0]].index.values # PDs[0] so happens to have all crime categories in it
Ncc = len(cr_index) 
mypos = posd(PDs, cr_index, NPDs, Ncc, xmin, xmax, ymin, ymax)
mypos.train(freq_pd, freq_add, N_add, df1)
# Training complete
# test data
nc = len(df2)
PDsTest = df2.PdDistrict.tolist()
print "%d test data crimes in total" % nc
PDsi = df2["PdDistrict"]
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
      pr = mypos.predict(PDsi[i], df2.Address[i], df2.X[i], df2.Y[i])
      csvw.writerow(pr)
      if i % 50000 == 0:
         print "i = %d" % i

