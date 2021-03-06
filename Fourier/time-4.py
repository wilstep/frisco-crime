import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from datetime import datetime
from fourier import Fourier

twopi = 2.0 * math.pi
t0 = np.datetime64("2003-01-06") # 00:00 hours, Monday   
swk = 7.0 * 24.0 * 3600.0 # seconds per week

## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
#vt = df[df.Category == 'DRUNKENNESS'] # Drunk Data
vt = df[df.Category == 'LARCENY/THEFT'] # Drunk Data
mydate = vt['Dates']
nd = len(mydate)
print "%d crimes in total" % nd
tw = np.zeros(nd) # array of times in weeks 
ni = np.linspace(0, nd-1, nd, dtype=np.int)
mydate.index = ni
for i in range (0, nd):
   tw[i] = (mydate[i]-t0).total_seconds()
   tw[i] /= swk
tw = tw[tw <= 100.0] # only times in the first 100 weeks
tw = tw % 1
tw = tw * 7
nc = len(tw)
# Now get Fourier curve
nw = 336   
nb = 84
t = np.linspace(0.0, 1.0, nw)
MyF = Fourier(645.0, 16, 52.1775, 12, 1.0, 6, 1.0/7.0, 8, df, t0)
MyF.compute()
fv = np.empty(nw)
#fv.fill(2700.0)
for i in range(0, nw):
   fv[i] = 0.0
   for j in range(0, 50):
      #fv[i] = MyF.f(t[i], 'DRUNKENNESS')
      fv[i] += MyF.f(t[i] + j * 2.0, 'LARCENY/THEFT')
   fv[i] *= nc + 0.0  # nc - number of crimes
   fv[i] /= 50.0 * nb # nb - number of histogram bins
# Got Fourier curve
t = t * 7

plt.hist(tw, bins=nb) # 645 weeks but we only have alternating weeks
plt.plot(t, fv, linewidth=2.5)
plt.title("Larceny/Theft")
plt.xlabel("Days")
plt.ylabel("50 times average Larceny/Theft arrests per 2 hours")
plt.savefig('time-4.png')
plt.show()

