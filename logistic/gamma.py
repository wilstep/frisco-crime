import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from math import log
import scipy.optimize as opt

z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)

# custom split, made so it doesn't split through same time stamps
# i.e. all samples with the same time stamp remain in the same data set
def mysplit(df, seed=6):
   date_list = df.Dates.unique()
   nds = len(date_list)
   np.random.seed(seed)
   lst = np.random.randint(100, size=nds) # every date given a number 0 - 99
   date_list = pd.Series(index=date_list, data=lst)
   date_list = date_list.apply(lambda x: True if x < 50 else False)
   df1 = df[df.Dates.apply(lambda x: date_list[x])]
   df2 = df[df.Dates.apply(lambda x: not date_list[x])]
   return df1, df2

#split data using custom date splitter
df1, df2 = mysplit(df, 1)

addys = df2.Address.unique()
addys.sort()
na = len(addys)
#fx = np.empty([2,na])
n = nl = 0
cats = df.Category.unique()
cats.sort()
icats = pd.Series(index=cats, data=range(len(cats)))
#cat = 'LARCENY/THEFT'
#cat = 'OTHER OFFENSES'
#cat = 'KIDNAPPING'
#cat = 'MISSING PERSON'
#print cat


#Ensure a > 0 and b > a
def ftran(ar):
   a, b = ar
   a = a * a + 0.001
   b = a + b * b + 0.001
   return (a, b)  

def mfit(ar):
   a, b = ftran(ar)
   print "a = %f, b = %f, pr = %f," % (a, b, a/b),
   ary[:,4] = (ary[:,1]+a) / (ary[:,0]+b)
   ary[:,5] = 1.0 - ary[:,4]
   ary[:,6] = np.log(ary[:,4])
   ary[:,7] = np.log(ary[:,5])
   ary[:,8] = -ary[:,3] * ary[:,6] - (ary[:,2] - ary[:,3]) * ary[:,7]
   Sp = ary[:,8].sum()
   Sp /= float(len(df2))
   #print "a = %f, b = %f, pr = %f, Sp = %f" % (a, b, a/b, Sp)
   print "Sp = %f" % Sp
   return Sp



i = 0
ncats = len(cats)
catnh = np.empty([na, 2*ncats])
ary = np.empty([na, 9])

for addy, row in df2.groupby('Address'):
   dft = df1[df1.Address==addy]
   ary[i][0] = len(dft) # n df1
   ary[i][2] = len(row) # n df2
   for j in range(ncats):
      cat = cats[j]
      j2 = 2 * j
      catnh[i][j2] = len(dft[dft.Category == cat]) 
      catnh[i][j2+1] = len(row[row.Category == cat])
   i+=1
   if i%500 == 0: print i, na, addy

#ary[][1] is h1
#ary[][3] is h2


dfpars = pd.DataFrame(index=cats)

#Sp = mfit([2.0, 5.0])
#print "Sp =", Sp
for ix in range(len(cats)):
   i2 = ix * 2
   ary[:,1] = catnh[:,i2]
   ary[:,3] = catnh[:,i2+1]
   a0 = 2.0 
   a1 = 5.0
   sol = opt.fmin_bfgs(mfit,(a0, a1))
   #print "sol", sol
   a, b = ftran(sol)
   cat = cats[ix]
   dfpars.loc[cat,'a'] = a
   dfpars.loc[cat,'b'] = b
   print cat, "a = %f, b = %f" % (a, b)
   pr0 = float(len(df1[df1.Category==cat])) / float(len(df1))
   dfpars.loc[cat,'pr'] = a / b
   dfpars.loc[cat,'pr0'] = pr0
   lna = log(pr0)
   lnb = log(1.0-pr0)
   S = 0.0
   for i in range(na):
      h1 = ary[i][1]
      n1 = ary[i][0]
      h2 = ary[i][3]
      n2 = ary[i][2]
      if h1 <= 5 or h1==n1:
         S -= float(h2) * lna
         S -= float(n2-h2) * lnb
      else:
         pr = float(h1) / float(n1)
         S -= float(h2) * log(pr)
         S -= float(n2-h2) * log(1.0-pr)
   S /= float(len(df2))
   print "pr0 = %f, S = %f" % (pr0, S) 
dfpars.to_csv('params.csv')


