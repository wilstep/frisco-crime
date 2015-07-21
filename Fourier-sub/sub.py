import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from datetime import datetime
from fourier import Fourier

twopi = 2.0 * math.pi
t0 = np.datetime64("2003-01-01") # 00:00 hours, Wednesday   
swk = 7.0 * 24.0 * 3600.0 # seconds per week

## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
nc = len(df)
print "%d train data crimes in total" % nc
group = df.groupby('Category')
freq = group.size()
cr_index = freq.index.values
# Now train Fourier curve
MyF = Fourier(645.0, 16, 52.1775, 12, 1.0, 6, 1.0/7.0, 8, df, t0)
MyF.compute()
# Got Fourier data
z = zipfile.ZipFile('../test.csv.zip')
df = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)
nc = len(df)
print "%d test data crimes in total" % nc
dates = df.Dates
#mydate = vt['Dates']
#nd = len(mydate)
tw = np.zeros(nc) # array of times in weeks 
ni = np.linspace(0, nc-1, nc, dtype=np.int)
dates.index = ni
for i in range (0, nc):
   tw[i] = (dates[i]-t0).total_seconds()
   tw[i] /= swk
   tw[i] -= 1.0 # shift one week to line up with training data
pr = np.empty(39)
s1 = np.append(['Id'], [cr_index])
with open('out.csv','wb') as f_handle:
   csvw = csv.writer(f_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
   csvw.writerow(s1)
   for i in range(0,nc):
      s2 = '%d,' % i
      f_handle.write(s2)
      MyF.getProb(tw[i], pr)
      csvw.writerow(pr)
      if (i) % 50000 == 0:
         print "i = %d" % i


