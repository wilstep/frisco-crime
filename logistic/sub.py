import zipfile
import pandas as pd
import numpy as np
import gzip
from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from cdata import cdata

print "Pandas", pd.__version__
print "Numpy", np.version.version

z = zipfile.ZipFile('../train.csv.zip')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df1 = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'], date_parser=dateparse)
z = zipfile.ZipFile('../test.csv.zip')
df2 = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'], date_parser=dateparse)

#print df1.Dates.min(), df1.Dates.max()
#print df2.Dates.min(), df2.Dates.max()

print "\n{0} training samples and {1} test samples".format(len(df1), len(df2))
df1t = df1[df1.Y<40.0]
df2t = df2[df2.Y<40.0]
xmax = max(df1t.X.max(), df2t.X.max()) + 0.000000001
xmin = min(df1t.X.min(), df2t.X.min()) - 0.000000001
ymax = max(df1t.Y.max(), df2t.Y.max()) + 0.000000001
ymin = min(df1t.Y.min(), df2t.Y.min()) - 0.000000001


#ngrid = 35
#mdeg = 2 # polynomial degree

cats = df1.Category.unique()
cats.sort()

sl = []
def run(scale, scp, scp2, ngrid, addymin, mdeg, C):
   MyCData = cdata(scp, scp2, mdeg, ngrid, xmax, xmin, ymax, ymin, scale)
   MyCData.train(df1, C=C)
   Ytest = MyCData.test(df2)
   dfout = pd.DataFrame(columns=cats, data=Ytest)
   with gzip.GzipFile('myout.csv.gz',mode='w',compresslevel=9) as gzfile:
      dfout.to_csv(gzfile, index_label='Id')

ngridm = 3
naddym = 1
nc = 40
#sc = 0.022
sc = 0.02
scp = 0.25
scp2 = 1.5
C = 0.25
run(sc, scp, scp2, 30, 4, 5, C)

   

