# Fourier time series
# flat propabilities in terms of position

import zipfile
import pandas as pd
import numpy as np
import csv
import math

from datetime import datetime

# approximate a year as 52.1775 weeks
swk = 7.0 * 24.0 * 3600.0 # average seconds per week 7 * 24 * 3600


Nfc = 20 # total number of frequencies
an = np.zeros(Nfc) # all crimes
bn = np.zeros(Nfc)

# weeks - period, nhm - number of harmonics
def cFourier(weeks, nhm):
   syr = swk * weeks
   mydate = df['Dates']
   #rdate = rdate[::-1]
   tcrimes = freq.sum()
   twopi = 2.0 * math.pi
   print "period is %f weeks" % weeks
   an[:] = 0.0 
   for i in range(0,tcrimes): # loop over all crime incidents
      t = (mydate[i]-t0).total_seconds()
      wks = (t / syr) % 1
      for j in range(0,nhm): # loop over the harmonics
         dj = j + 1.0
         arg = twopi * wks * dj
         cost = math.cos(arg)
         sint = math.sin(arg)
         an[j] += cost
         bn[j] += sint
   tcr = tcrimes + 0.0
   tcr *= 0.5
   print "i, an, bn, cn, al, bl, cl"
   for i in range(0, nhm):
      an[i] /= tcr
      bn[i] /= tcr
      cn = math.sqrt(an[i] * an[i] + bn[i] * bn[i])
      print "harmonic %d, cn = %f" %(i+1, cn)
   print "\n\n"

#oFourier(645.0, 8, 52.1775, 12, 1.0, 6, 1.0/7.0, 6)
#def oFourier(w1, h1, w2, h2, w3, h3, w4, h4):
   

## read file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)

# compute time range
t0 = np.datetime64("2003-01-01")
tl = np.datetime64("2015-05-13")
t0 = (df['Dates'][0] - t0).total_seconds()
tl = (df['Dates'][0] - tl).total_seconds()
diff = t0 - tl
tw = (diff + 0.0) / (3600.0 * 24.0 * 7.0)
print "full duration of data runs over %f weeks" % tw  # total weeks data spans
t0 = np.datetime64("2003-01-01")

crime_category = df['Category']
group = df.groupby('Category')
freq = group.size()
print
cr_index = freq.index.values
Nc = len(cr_index)
cr_a_index = pd.DataFrame(data = np.arange(Nc, dtype=np.int), index = cr_index) 
cr_a_index = cr_a_index[0] # this now holds the index for each crime type
#for i in range(0, Nc):
#   print cr_index[i]
#   print cr_a_index[cr_index[i]] #cr_a_index[cr_index[i]]
#cr_index = pd.DataFrame(index = c    # pdf.index.values # now a Numpy array
#cr_index = pd.DataFrame(index = c    # pdf.index.values # now a Numpy array
#print 'end of list'
pdf = freq
pdf = pdf.astype(float)
sumt = pdf.sum()
pdf = np.divide(pdf, sumt)
print t0
#for i in range(0,25):
#   #print "%-30s %d %f" % (cr_index[i], freq[cr_index[i]], pdf[cr_index[i]])
#   print df['Dates'][i]
#   t = (df['Dates'][i]-t0).total_seconds()
cFourier(645.0, 20)   # all data
cFourier(52.1775, 20) # period a year
cFourier(1.0, 20) # a week
cFourier(1.0/7.0, 6) # a day  
#print freq

   #print pd.DatetimeIndex(df['Dates'][i])
#s1 = np.append(['Id'], [cr_index])
#with open('out.csv','wb') as f_handle:
#   csvw = csv.writer(f_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#   csvw.writerow(s1)
#   for i in range(0,884262):
#      s2 = '%d,' % i
#      f_handle.write(s2)
#      csvw.writerow(pdf)

