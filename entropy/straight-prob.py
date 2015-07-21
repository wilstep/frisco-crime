# Straight probability for each crime
# independent of time and space, score 2.68016, place 52 of 82

import zipfile
import pandas as pd
import numpy as np
import csv

z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)

crime_category = df['Category']
group = df.groupby('Category')
freq = group.size()
print freq
print freq['WEAPON LAWS']
print freq['TREA']
cr_index = freq.index.values
Nc = len(cr_index)
pdf = freq
pdf = pdf.astype(float)
sumt = pdf.sum()
pdf = np.divide(pdf, sumt)
cr_index = pdf.index.values # now a Numpy array
print pdf.shape
print pdf.dtype
print
print
for i in range(Nc-10,Nc):
   print "%-30s %d %f" % (cr_index[i], freq[cr_index[i]], pdf[cr_index[i]])
s1 = np.append(['Id'], [cr_index])
with open('out.csv','wb') as f_handle:
   csvw = csv.writer(f_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
   csvw.writerow(s1)
   for i in range(0,884262):
      s2 = '%d,' % i
      f_handle.write(s2)
      csvw.writerow(pdf)

