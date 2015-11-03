import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from math import log
from math import exp
from math import sqrt
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from scipy.special import expit

#mdeg = 2 # order for x, y, t polynomial

# For probabilities, just put in ln(pi)

class cdata:
   """Data for use with logistic regression
      set all dud X & Y coords to zero after preprocessing the other data
      If X & Y coordinates are duds, and the address # is too small, then use PD probs"""
   def __init__(self, scp, scp2, mdeg, Ng, xmax, xmin, ymax, ymin, scale=0.05):
      self.Ng = Ng # linear number of grid points
      self.xmax = xmax
      self.xmin = xmin
      self.ymax = ymax
      self.ymin = ymin
      self.dx = (self.xmax - self.xmin) / float(self.Ng)
      self.dy = (self.ymax - self.ymin) / float(self.Ng)
      self.mdeg = mdeg
      self.Npca = 95 # No. for principle component analysis
      self.pmax = 0.95
      self.lmax = log(self.pmax / (1.0 - self.pmax))
      self.scale = scale
      self.scp = scp
      self.scp2 = scp2

   def train(self, df1, C=1.1):
      #print df1.Dates.min(), df1.Dates.max()
      self.cr_index = df1.Category.unique()
      self.cr_index.sort() # now have array of crime categories
      self.Ncc = len(self.cr_index) 
      print "Ncc = %d, %d x %d grid" % (self.Ncc, self.Ng, self.Ng)
      print "poly order = %d, Npca = %d" % (self.mdeg, self.Npca)
      self.iccl = pd.Series(data = np.arange(self.Ncc, dtype=np.int), index = self.cr_index) 
      self.__AddyMask(df1) # Build up address table
      # Now have built up what we need, test routine to mirror from here
      atr = self.__munge_1(df1, 'training')      
      self.sc = self.scale / sqrt(C)
      self.scaler_1 = StandardScaler()
      self.C = C
      X2 = self.__munge_2(df1, 'training', True)
      nfeat = len(X2[0])
      #self.scaler_1.fit(atr)
      #atr = self.scaler_1.transform(atr)
      print "Now training logistic model with %d features" % (nfeat+1)
      self.lgs = {}
      for cat in self.cr_index:
         print cat
         ds = pd.Series(data=df1.Category)
         ds = ds.apply(lambda x: 1 if x==cat else 0)
         self.lgs[cat] = LogisticRegression(C=self.C)
         #X = np.transpose(X)
         #X = X.reshape(())
         x = atr[cat].as_matrix()
         #x -= x.mean()
         X = np.empty([len(df1),1])
         X[:,0] = x
         X *= self.sc
         X = np.concatenate((X, X2), axis=1)
         self.lgs[cat].fit(X, ds)
      print "Training complete!"
      
   def test(self, df):
      atest = self.__munge_1(df, 'testing')
      X2 = self.__munge_2(df, 'testing', False)
      Xp = pd.DataFrame(columns=self.cr_index)
      print "predicting probabilities"
      for cat in self.cr_index:
         x = atest[cat].as_matrix()
         #x -= x.mean()
         X = np.empty([len(df),1])
         X[:,0] = x
         X *= self.sc
         X = np.concatenate((X, X2), axis=1)
         pr = self.lgs[cat].predict_proba(X)
         Xp[cat] = pr[:,1]
      Xp = Xp.as_matrix()
      for ix in range(len(df)):
         sm = Xp[ix].sum()
         Xp[ix] /= sm   
      return Xp
   
   def __munge_1(self, df, mstr):
      print "phase 1 of data preparation for logistic regression on the %s data" % mstr
      nrows = len(df)
      addy_N = np.zeros([nrows, self.Ncc]) # addresses
      for ix, row in df.iterrows():
         addy = row.Address   
         if addy in self.addys:
            addy_N[ix] = np.copy(self.addy_N[self.addys[addy]])
         else: addy_N[ix] = np.copy(self.ln0)
      dfa = pd.DataFrame(data=addy_N, columns=self.cr_index)
      return dfa
   
   def __munge_2(self, df, mstr, tst):
      print "phase 2 of data preparation for logistic regresion on the %s data" % mstr   
      dft = pd.get_dummies(df['PdDistrict'])
      ncpos = len(dft.columns)
      nrows = len(df)
      agb = df.groupby('Address').size()
      dft['Naddy'] = df.Address.apply(lambda x: agb[x])
      repeat_gb = df.groupby(['Address', 'Dates']).size()
      dupli = np.zeros([4,nrows])
      # 2-3, 4-6, 7-10, 11- 
      for ix, row in df.iterrows():
         addy = row.Address
         date = row.Dates
         #if addy in self.repeat_gb and date in self.repeat_gb[addy]:
         cnt = repeat_gb[addy][date]
         if cnt >= 11:
            dupli[0][ix] = 1
         elif cnt >= 7:
            dupli[1][ix] = 1        
         elif cnt >= 4:
            dupli[2][ix] = 1  
         elif cnt >= 2:
            dupli[3][ix] = 1 
      dft.loc[:, 'dupli0'] = dupli[0]
      dft.loc[:, 'dupli1'] = dupli[1]
      dft.loc[:, 'dupli2'] = dupli[2]
      dft.loc[:, 'dupli3'] = dupli[3]    
      dft['corner'] = df.Address.apply(lambda addy: 1 if ('/' in addy) else 0)  
      dfp = pd.DataFrame()
      dfp.loc[:,'X'] = df.X
      dfp.loc[:,'Y'] = df.Y
      dfp.loc[:,'Dud'] = dfp.Y.apply(lambda y: 1 if y > 40.0 else 0)
      xm = dfp.X[dfp.Dud==0].mean()
      ym = dfp.Y[dfp.Dud==0].mean()
      dfp.loc[:,'X'] = dfp.apply(lambda r: r.X - xm if r.Dud==0 else 0.0, axis=1)
      dfp.loc[:,'Y'] = dfp.apply(lambda r: r.Y - ym if r.Dud==0 else 0.0, axis=1)
      del dfp['Dud']
      poly = PolynomialFeatures(degree=self.mdeg, include_bias=False)
      dfp = poly.fit_transform(dfp)
      nc = len(dfp[0])
      ncpos += nc
      dftemp = self.__temporal(df)
      Xp1 = np.concatenate((dfp, dftemp.as_matrix(), dft.as_matrix()), axis=1)
      #Xp1 = np.concatenate((dft.as_matrix(), dftemp.as_matrix()), axis=1)
      if tst: self.scaler_1.fit(Xp1)
      Xp1 = self.scaler_1.transform(Xp1)
      sf = self.scp / sqrt(self.C)
      nc2 = nc + 10
      Xp1[:,0:nc] *= sf
      sf = self.scp2 / sqrt(self.C)
      Xp1[:,nc:nc2] *= sf 
      Xp = Xp1
      print "phase 2 done, now have %d features" % len(Xp[0])
      return Xp 

   def __temporal(self, df):
      print 'Munging temporal features'
      t0 = np.datetime64("2003-01-01") # 00:00 hours, Wednesday
      tl = np.datetime64("2015-05-14")
      tspan = tl - t0
      tspan = tspan.item().total_seconds() # total hrs spanned
      myt = pd.DataFrame(index=df.index)
      myt['t'] = df.Dates.apply(lambda x: (x-t0).total_seconds())
      tav = myt.t.mean()
      myt['t'] = myt.t.apply(lambda x: 2.0*(x-tav)/tspan)
      myt['t2'] = myt.t.apply(lambda x: x*x)
      myt['t3'] = myt.apply(lambda r: r.t*r.t2, axis=1)
      myt['t4'] = myt.t2.apply(lambda x: x*x)
      myt['t5'] = myt.apply(lambda r: r.t4*r.t, axis=1)
      myt['td'] = df.Dates.apply(lambda x: ((x.hour+18)%24)*60 + x.minute)
      tav = myt.td.mean()
      myt['td'] = myt.td.apply(lambda x: (x-tav)/720.0)
      myt['td2'] = myt.td.apply(lambda x: x*x)
      myt['td3'] = myt.apply(lambda r: r.td*r.td2, axis=1)
      myt['td4'] = myt.td2.apply(lambda x: x*x)
      myt['td5'] = myt.apply(lambda r: r.td4*r.td, axis=1)
      myt['we'] = df.Dates.apply(lambda x: x.hour + x.day * 24) # week end
      # Friday = 4 * 24 = 96 & Sunday = 6 * 24 = 144, 8pm = 20 so 116 & 164
      myt['we'] = myt.we.apply(lambda x: 1 if (x >= 116 and x <= 164) else 0) 
      myt['ew'] = df.Dates.apply(lambda x: x.hour + x.day * 24)
      myt['ew'] = myt.we.apply(lambda x: 1 if (x >= 68 and x <= 116) else 0) 
      # dummy variable for those who play up on the weekend
      myt['drift'] = df.Dates.apply(lambda x: 3.0 * ((x-t0).total_seconds()))
      myt['drift'] = myt.drift.apply(lambda x: int(x / tspan))
      dfdummy1 = pd.get_dummies(myt['drift'])
      del myt['drift']
      dfdummy1.columns = ['d1', 'd2', 'd3']
      myt['night'] = df.Dates.apply(lambda x: 1 if (x.hour < 7 or x.hour > 23) else 0)
      myt['day'] = df.Dates.apply(lambda x: 1 if (x.hour >= 7 and x.hour < 17) else 0)
      myt['season'] = df.Dates.apply(lambda x: ((x.month+1) / 3)%4)
      dfdummy2 = pd.get_dummies(myt['season'])
      del myt['season']
      myt = pd.concat([myt, dfdummy1, dfdummy2], axis=1)
      print "number of temporal features", len(myt.columns)
      return myt   
     
   
   def __AddyMask(self, df):
      """ Make a bool array with 39 columns for each address, 
          that is True for the crimes > Naddmin, only has addresses
          with at least one true entry """
      dfp = pd.read_csv('params.csv', index_col=0)
      self.GlobalPr = np.empty(self.Ncc)
      gpr = df.groupby('Category').size()
      nsamp = len(df)
      sp = 0.0
      for ix in range(self.Ncc): 
         cat = self.cr_index[ix]
         #pr = float(gpr[cat]) / float(nsamp)
         pr = dfp.a[cat] / dfp.b[cat]
         sp += pr
         self.GlobalPr[ix] = pr
      self.ln0 = np.empty(self.Ncc)
      for ix in range(self.Ncc):
         pr = self.GlobalPr[ix] / sp
         self.ln0[ix] = log(pr/(1.0-pr))
      self.addys = df.Address.unique() 
      self.addys.sort()
      naddy = len(self.addys)
      self.addys = pd.Series(data = np.arange(naddy, dtype=np.int), index = self.addys)   
      print "Making table for %d Addresses" % naddy
      addyg = df.groupby('Address') 
      self.addy_N = np.empty([naddy, self.Ncc])
      self.addy_N.fill(0.0)
      ia = 0
      for addy, data in addyg:
         ia = self.addys[addy]
         ccnt = data.groupby('Category').size() # histo of crimes for this row
         csum = ccnt.sum()
         #self.addy_N[ia][self.Ncc] = log(float(csum))
         ja = 0
         sp = 0.0
         for key in self.cr_index:
            if key in ccnt: # have this crime at this address
               na = ccnt[key]
               pn = (dfp.a[key] + float(na)) / (dfp.b[key] + float(csum))
               sp += pn
               self.addy_N[ia][ja] = pn
            else: # no crimes of this cat at this address
               pn = self.GlobalPr[ja]
               sp += pn
               self.addy_N[ia][ja] = pn
            ja += 1
         ja = 0
         for key in self.cr_index:
            pn = self.addy_N[ia][ja] / sp
            self.addy_N[ia][ja] = log(pn/(1.0-pn))  
            ja += 1            
      pass # end for loop

   # Duds -120.5, 90.0
   #  too small, too large
   def __gd(self, x, y):
      if y >= 40.0: return int(0)
      xt = x - self.xmin
      xt = int(xt / self.dx)
      yt = y - self.ymin
      yt = int(yt / self.dy)
      i = xt + yt * self.Ng
      return int(i)  



