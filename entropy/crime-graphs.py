import zipfile
import pandas as pd
#import numpy as np
#import requests

#import seaborn as sns
import matplotlib.pyplot as plt

z = zipfile.ZipFile('../train.csv.zip')
print(z)

df = pd.read_csv(z.open('train.csv'), parse_dates=True, index_col=0)

by_address = df.groupby('Address')
by_category = df.groupby('Category')
dfb = df[df['Address'] == "800 Block of BRYANT ST"]
Bryant_category = dfb.groupby('Category')

print "length Bryant St %d" % len(Bryant_category)

addr_freq = by_address.size()
cate_freq = by_category.size()
bcate_freq = Bryant_category.size()

crime_loc  = addr_freq.sort(ascending=False, inplace=False)
print "number of locationssss %d" % len(crime_loc)

crime_loc  = addr_freq.sort(ascending=False, inplace=False)[0:20]
crime_type = cate_freq.sort(ascending=False, inplace=False)
bcrime_type = bcate_freq.sort(ascending=False, inplace=False)

for i in range(0,10):
   print "%d total %d, Bryant St %d" % (i+1, crime_type[i], bcrime_type[i]) 

crime_loc = crime_loc[::-1]
crime_type = crime_type[::-1]
bcrime_type = bcrime_type[::-1]

#fig, ax = plt.subplots(figsize=(12,8))
#crime_loc.plot(kind='bar', title='Top Crime Zones', ax=ax)
#crime_loc.plot(kind='barh', title='Top Crime Zones', ax=ax)
#plt.tight_layout()
#plt.savefig('Top Crime Zones.png', figsize=(8,12))

fig, ax = plt.subplots(figsize=(12,8))
crime_type.plot(kind='barh', title='Top Crimes Committed', ax=ax)
plt.tight_layout()
plt.savefig('crime-histo.png', figsize=(8,12))

#fig, ax = plt.subplots(figsize=(12,8))
#bcrime_type.plot(kind='barh', title='Top Crimes Committed', ax=ax)
#plt.tight_layout()
#plt.savefig('Bryant Top Crime Committed.png', figsize=(8,12))
