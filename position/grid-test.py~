import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math


twopi = 2.0 * math.pi
t0 = np.datetime64("2003-01-01") # 00:00 hours, Wednesday   
swk = 7.0 * 24.0 * 3600.0 # seconds per week

## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
#df = df[df.X > -122.0]
df1 = df[df.Y > 38.0]
X = df1.X
Y = df1.Y
nX = len(X)
print "Training Data"
print nX
for i in range(nX):
   print "X = %f, Y = %f" % (X.iloc[i], Y.iloc[i]),
   print df1.PdDistrict.iloc[i]
df1 = df[df.Y < 38.0]
X = df1.X
Y = df1.Y
nX = len(X)
xa = X.mean()
ya = Y.mean()
print "<X> = %f, <Y> = %f" % (xa, ya)
print "Xmax = %f, Ymax = %f" % (X.max(), Y.max())
print "Xmin = %f, Ymin = %f" % (X.min(), Y.min())
#Check out the test data
z = zipfile.ZipFile('../test.csv.zip')
df = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)
df1 = df[df.Y > 38.0]
X = df1.X
Y = df1.Y
nX = len(X)
print "Test Data"
print nX
for i in range(nX):
   print "X = %f, Y = %f" % (X.iloc[i], Y.iloc[i]),
   print df1.PdDistrict.iloc[i]
df1 = df[df.Y < 38.0]
X = df1.X
Y = df1.Y
nX = len(X)
xa = X.mean()
ya = Y.mean()
print "<X> = %f, <Y> = %f" % (xa, ya)
print "Xmax = %f, Ymax = %f" % (X.max(), Y.max())
print "Xmin = %f, Ymin = %f" % (X.min(), Y.min())

