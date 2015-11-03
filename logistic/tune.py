import zipfile
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from cdata import cdata

print "Pandas", pd.__version__
print "Numpy", np.version.version

z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
#df = df[df.index<200000]
#df1, df2 = train_test_split(df, train_size = 0.9, random_state = 42)

# custom split, made so it doesn't split through same time stamps
# i.e. all samples with the same time stamp remain in the same data set
def mysplit(df):
   date_list = df.Dates.unique()
   nds = len(date_list)
   np.random.seed(7)
   lst = np.random.randint(100, size=nds)
   date_list = pd.Series(index=date_list, data=lst)
   date_list = date_list.apply(lambda x: True if x < 50 else False)
   df1 = df[df.Dates.apply(lambda x: date_list[x])]
   df2 = df[df.Dates.apply(lambda x: not date_list[x])]
   return df1, df2
#df1, df2 = train_test_split(df, train_size = 0.5, random_state = 42)

#split data using custom date splitter
df1, df2 = mysplit(df)


df1.index = range(len(df1))
df2.index = range(len(df2))
print "\n{0} training samples and {1} test samples".format(len(df1), len(df2))
df1t = df1[df1.Y<40.0]
df2t = df2[df2.Y<40.0]
xmax = max(df1t.X.max(), df2t.X.max()) + 0.000000001
xmin = min(df1t.X.min(), df2t.X.min()) - 0.000000001
ymax = max(df1t.Y.max(), df2t.Y.max()) + 0.000000001
ymin = min(df1t.Y.min(), df2t.Y.min()) - 0.000000001

l1 = df1.Category.unique()
l1.sort()
l2 = df2.Category.unique()
l2.sort()
def Ent(y, Xp):
   dfXp = pd.DataFrame(data=Xp,columns=l1)
   for col in l1:
      if col not in l2: del dfXp[col]
   return log_loss(y, dfXp.as_matrix())  


#ngrid = 35
#mdeg = 2 # polynomial degree

sl = []
def run(scale, scp, scp2, ngrid, addymin, mdeg, C):
   MyCData = cdata(scp, scp2, mdeg, ngrid, xmax, xmin, ymax, ymin, scale)
   MyCData.train(df1, C=C)
   Ytest = MyCData.test(df2)
   y = df2.Category.as_matrix()
   s = "scale = %f, C = %f, scp = %f, scp2 = %f, order = %d, logloss = %f" % (scale, C, scp, scp2, mdeg, Ent(y, Ytest))
   #print s
   sl.append(s)

ngridm = 3
naddym = 1
nc = 40
#sc = 0.022
sc = 0.02
scp = 0.25
scp2 = 1.5
C = 0.25
for j in range(nc):
   # scaling, scp, grid, addy, order, regurlise
   run(sc, scp, scp2, 30, 4, 5, C)
   #scp2 += 0.1
   #C += 0.05
   sc += 0.002
   print '\n'
   for j in range(len(sl)): print sl[j]
   print
   

