import zipfile
import pandas as pd
import numpy as np
#import requests

#import seaborn as sns
import matplotlib.pyplot as plt

z = zipfile.ZipFile('../train.csv.zip')
print(z)

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)

by_address = df.groupby('Address')
by_category = df.groupby('Category')
N = len(df)
#dfb = df[df['Address'] == "800 Block of BRYANT ST"]
#Bryant_category = dfb.groupby('Category')

#print "length Bryant St %d" % len(Bryant_category)

addr_freq = by_address.size()
cate_freq = by_category.size()
#bcate_freq = Bryant_category.size()

crime_loc  = addr_freq.sort(ascending=False, inplace=False)
print "number of addresses %d" % len(crime_loc) 

tot = 0
for i in range(0,1000):
   if crime_loc.values[i] >= 500:
      j = i + 1
      print j,
      print ", Address: ",
      print crime_loc.index[i],
      print ", # of crimes: ",
      print crime_loc[i]
      tot += crime_loc.values[i]
print "sub-total = %d of %d crimes" % (tot, N) 


