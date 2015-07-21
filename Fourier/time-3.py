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
swk = 7.0 * 24.0 * 3600.0 # seconds per week

## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
vt = df[df.Category == 'VEHICLE THEFT'] # Vehicle Theft Data
mydate = vt['Dates']
nd = len(mydate)
print "%d crimes in total" % nd
tw = np.zeros(nd) # array of times in weeks 
ni = np.linspace(0, nd-1, nd, dtype=np.int)
mydate.index = ni
for i in range (0, nd):
   tw[i] = (mydate[i]-t0).total_seconds()
   tw[i] /= swk
# Now get Fourier curve
nw = 323   
t = np.linspace(0.5, 644.5, nw)
#MyF = Fourier(645.0, 20, 52.1775, 12, 1.0, 6, 1.0/7.0, 6, df)
MyF = Fourier(645.0, 16, 52.1775, 12, 1.0, 0, 1.0/7.0, 0, df, t0)
MyF.compute()
fv = np.empty(nw)
#fv.fill(2700.0)
for i in range(0, nw):
   fv[i] = MyF.f(t[i], 'VEHICLE THEFT')
   fv[i] *= nd + 0.0
   fv[i] /= 322.0
# Got Fourier curve

plt.hist(tw, bins=322) # 645 weeks but we only have alternating weeks
plt.plot(t, fv, linewidth=2.5)
plt.title("Vehicle Theft History")
plt.xlabel("Weeks")
plt.ylabel("Thefts per week")
plt.savefig('time-3.png')
plt.show()

