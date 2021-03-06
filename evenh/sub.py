import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import copy
import sklearn
from datetime import datetime
from sklearn.cross_validation import train_test_split
from grid import grid

## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df1 = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
# test data
z = zipfile.ZipFile('../test.csv.zip')
df2 = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)
# have loaded all the data

#print sklearn.__version__
#df1, df2 = train_test_split(df, train_size = 0.95)

xmax = max(df1[df1.Y < 40.0].X.max(), df2[df2.Y < 40.0].X.max()) + 0.0000001
xmin = min(df1[df1.Y < 40.0].X.min(), df2[df2.Y < 40.0].X.min()) - 0.0000001
ymax = max(df1[df1.Y < 40.0].Y.max(), df2[df2.Y < 40.0].Y.max()) + 0.0000001
ymin = min(df1[df1.Y < 40.0].Y.min(), df2[df2.Y < 40.0].Y.min()) - 0.0000001

cr_index = df1.Category.unique()
cr_index.sort() # now have array of crime categories
Ncc = len(cr_index) 
iccl = pd.DataFrame(data = np.arange(Ncc, dtype=np.int), index = cr_index)
iccl = iccl[0] # cc index, put cc in, get int out

print "Numbers, df1 = %d, df2 = %d" % (len(df1), len(df2))
nc = len(df2)
s = np.empty(nc)
se2 = np.empty(nc)
print 
print "%d test data crimes in total" % nc

def Entropy(pr):
   n = len(pr)
   x = 0.0
   for i in range(n):
      #xi = float(pr[i])
      if pr[i] > 0.0:
         x -= pr[i] * math.log(pr[i])
   return x 

s1 = np.append(['Id'], [cr_index])

def Run(n):
   Ngl1 = 70 + 4 * n
   #Ngl2 = 10 + 2 * n
   mygrid = grid(xmin, xmax, ymin, ymax, Ngl1, 18)
   mygrid.train(df1, 6, 7)
   print
   print "Run(n), Ngl1 = %d" % Ngl1
   #print "Run(n), Ngl2 = %d" % Ngl2
   #print "Run(n), n = %d" % n
   with open('out.csv','wb') as f_handle:
      csvw = csv.writer(f_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
      csvw.writerow(s1)
      i = 0
      #for i in range(nc):
      for idx, row in df2.iterrows():
         s2 = '%d,' % i
         f_handle.write(s2)
         pr = mygrid.predict(df2.Address[idx], df2.X[idx], df2.Y[idx], df2.PdDistrict[idx])
         crm = df2.Category[idx]
         j = iccl[crm]         
         if pr[j] > 1.0E-15: s[i] = -math.log(pr[j])
         else: s[i] = -math.log(1.0E-15)
         se2[i] = Entropy(pr)
         csvw.writerow(pr)
         if i % 5000 == 0:
            print "i = %d, s = %f, se2 = %f" % (i, s[:i+1].mean(), se2[:i+1].mean())
         i += 1
   #print "n = %d, S = %f, Se2 = %f" % (n, s.mean(), se2.mean())
   print "Ngl1 = %d, S = %f, Se2 = %f" % (Ngl1, s.mean(), se2.mean())
   #print "Ngl2 = %d, S = %f, Se2 = %f" % (Ngl2, s.mean(), se2.mean())
   print "\n\n"

#for i in range(0, 40):
#   Run(i)

with open('out.csv','wb') as f_handle:
   csvw = csv.writer(f_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
   csvw.writerow(s1)
   mygrid = grid(xmin, xmax, ymin, ymax, 80, 18)
   mygrid.train(df1, 6, 7)
   i = 0
   for idx, row in df2.iterrows():
      s2 = '%d,' % i
      f_handle.write(s2)
      pr = mygrid.predict(df2.Address[idx], df2.X[idx], df2.Y[idx], df2.PdDistrict[idx])
      csvw.writerow(pr)
      if i % 5000 == 0:
         print "i = %d" % i
      i += 1


