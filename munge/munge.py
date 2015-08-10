import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import copy
from datetime import datetime

Amin = 1000 # minimum counts to use address
Gmin = 1500 # min count to use grid
Ngl1 = 40 # Ngl x Ngl grid
Ntgl1 = Ngl1 * Ngl1
Ngl2 = 12 # Ngl x Ngl grid
Ntgl2 = Ngl1 * Ngl1


## read training file
z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df1 = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
# test data
z = zipfile.ZipFile('../test.csv.zip')
df2 = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)
# have loaded all the data


xmax = max(df1[df1.Y < 40.0].X.max(), df2[df2.Y < 40.0].X.max()) + 0.0000001
xmin = min(df1[df1.Y < 40.0].X.min(), df2[df2.Y < 40.0].X.min()) - 0.0000001
ymax = max(df1[df1.Y < 40.0].Y.max(), df2[df2.Y < 40.0].Y.max()) + 0.0000001
ymin = min(df1[df1.Y < 40.0].Y.min(), df2[df2.Y < 40.0].Y.min()) - 0.0000001
xl = xmax - xmin # total grid length
yl = ymax - ymin
x0 = xmin
y0 = ymin
dx1 = xl / (Ngl1 + 0.0)
dy1 = yl / (Ngl1 + 0.0)
dx2 = xl / (Ngl2 + 0.0)
dy2 = yl / (Ngl2 + 0.0)
print "xmin = %f, xmax = %f" % (xmin, xmax)
print "ymin = %f, ymax = %f" % (ymin, ymax)

def fcell(x, y, dx, dy, Ngl):
   if y > 40.0:
      return -1
   ix = int((x - x0) / dx)
   iy = int((y - y0) / dy)
   ig = iy * Ngl + ix  
   return ig

def Munge(df, PD):
   #df = dfi.copy() # Make deep copy of data frame
   df["grid1"] = df.apply(lambda x: fcell(x['X'], x['Y'], dx1, dy1, Ngl1), axis=1)
   df["grid2"] = df.apply(lambda x: fcell(x['X'], x['Y'], dx2, dy2, Ngl2), axis=1)
   # Now find addresses with at least Amin crimes
   group = df.groupby("Address")
   freq = group.size()
   df["cnt"] = df.Address.apply(lambda addy: freq[addy])
   df.Address = df.Address.where(df.cnt >= Amin, "NFA")
   # grid 1 with at least Gmin
   group = df.groupby("grid1")
   freq = group.size()
   df["cnt"] = df.grid1.apply(lambda grid: freq[grid])
   #df.grid1[df.cnt <= Gmin] = -1
   df.grid1 = df.grid1.where(df.cnt >= Gmin, -1) 
   # grid 2 with at least Gmin
   group = df.groupby("grid2")
   freq = group.size()
   df["cnt"] = df.grid2.apply(lambda grid: freq[grid])
   df.grid2 = df.grid2.where(df.cnt >= Gmin, -1)  
   if PD: return df[["PdDistrict", "Address", "grid1", "grid2", "Category"]]
   return df[["PdDistrict", "Address", "grid1", "grid2"]]

tdf = Munge(df1, True)
print "write mtrain.csv"
tdf.to_csv("mtrain.csv", index = False)
tdf = Munge(df2, False)
print "write mtest.csv"
tdf.to_csv("mtest.csv", index = False)

