import pandas as pd
import numpy as np
import math
import sklearn

df = pd.read_csv('ent.csv')


n = len(df) 
print "n = %d" % n
sav = np.empty(n)
for i in range(n):
   s = 0
   for j in range(39):
      col = df.columns[1+j]
      x = df[col][i]
      if x == 0: continue
      x = x * math.log(x)
      s += x
   if i % 1000 == 0: print "i = %d, s = %f" % (i, -s)
   sav[i] = -s
print "sav = %f" % sav.mean()
       
