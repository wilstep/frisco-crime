import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

twopi = 2.0 * math.pi
#t0 = np.datetime64("2003-01-01")

class Fourier:
   def __init__(self, w1, n1, w2, n2, w3, n3, w4, n4, df, t0):  
      self.w1 = w1
      self.n1 = n1
      self.w2 = w2
      self.n2 = n2
      self.w3 = w3
      self.n3 = n3
      self.w4 = w4
      self.n4 = n4     
      self.t0 = t0
      self.Nfc = n1 + n2 + n3 + n4
      self.an = np.zeros(self.Nfc) # sum of all crimes, numpy array
      self.bn = np.zeros(self.Nfc)
      self.al = np.zeros((39, self.Nfc)) # specific crimes, numpy array
      self.bl = np.zeros((39, self.Nfc))
      self.st = "I am Fourier"
      self.df = df
      group = self.df.groupby('Category')
      self.crime_category = self.df['Category'] # list of crime types
      self.freq = group.size()   # histogram of crime types
      self.cr_index = self.freq.index.values  
      self.tcrimes = self.freq.sum() # total number of crimes in training data
      self.Nc = len(self.cr_index)    # number of crime types
      self.cr_a_index = pd.DataFrame(data = np.arange(self.Nc, dtype=np.int), index = self.cr_index) 
      self.cr_a_index = self.cr_a_index[0] # this now holds the integer index for each crime catagory
   # compute the Fourier coefficients
   def compute(self):
      n = 0
      self.cFourier(self.w1, n, self.n1)
      n += self.n1
      self.cFourier(self.w2, n, self.n2)
      n += self.n2
      self.cFourier(self.w3, n, self.n3)
      n += self.n3
      self.cFourier(self.w4, n, self.n4)
   def graph(self):
      #t = np.arange(0.0, 650.0, 2.5)
      t = np.arange(0.0, 1.0, 0.01)
      n = len(t)
      fa = np.zeros(n)
      fb = np.zeros(n)
      fc = np.zeros(n)
      for i in range(0, n):
         fa[i] = self.fa(t[i])
         fb[i] = self.f(t[i], 'DRUNKENNESS')
         fc[i] = self.f(t[i], 'DRUG/NARCOTIC')
      fa = fa / fa
      fb = fb / fa
      fc = fc / fa
      fig, ax = plt.subplots(nrows=1, ncols=1)
      ax.plot(t, fa, 'k--', label='All Crimes')
      ax.plot(t, fb, 'k:', label='Drunkenness')
      ax.plot(t, fc, 'k', label='Drug/Narcotic')
      # Now add the legend with some customizations.
      legend = ax.legend(loc='upper right', shadow=True)
      # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
      frame = legend.get_frame()
      frame.set_facecolor('0.90')
      fig.savefig('period.png')
      #plt.savefig('period.png', t, self.f(t), figsize=(8,12))
   def fa(self, tw):
      self.sm = 1.0
      for i in range(0, self.n1):
         arg = twopi * tw * (i + 1.0) / self.w1
         self.sm += self.an[i] * math.cos(arg)
         self.sm += self.bn[i] * math.sin(arg)
      for i in range(0, self.n2):
         arg = twopi * tw * (i + 1.0) / self.w2
         j = i + self.n1
         self.sm += self.an[j] * math.cos(arg)
         self.sm += self.bn[j] * math.sin(arg)
      for i in range(0, self.n3):
         arg = twopi * tw * (i + 1.0) / self.w3
         j = i + self.n1 + self.n2
         self.sm += self.an[j] * math.cos(arg)
         self.sm += self.bn[j] * math.sin(arg)
      for i in range(0, self.n4):
         arg = twopi * tw * (i + 1.0) / self.w4
         j = i + self.n1 + self.n2 + self.n3
         self.sm += self.an[j] * math.cos(arg)
         self.sm += self.bn[j] * math.sin(arg) 
      return self.sm     
   def f(self, tw, crm):
      k = self.cr_a_index[crm]
      sm = 1.0
      for i in range(0, self.n1):
         arg = twopi * tw * (i + 1.0) / self.w1
         sm += self.al[k][i] * math.cos(arg)
         sm += self.bl[k][i] * math.sin(arg)
      for i in range(0, self.n2):
         arg = twopi * tw * (i + 1.0) / self.w2
         j = i + self.n1
         sm += self.al[k][j] * math.cos(arg)
         sm += self.bl[k][j] * math.sin(arg)
      for i in range(0, self.n3):
         arg = twopi * tw * (i + 1.0) / self.w3
         j = i + self.n1 + self.n2
         sm += self.al[k][j] * math.cos(arg)
         sm += self.bl[k][j] * math.sin(arg)
      for i in range(0, self.n4):
         arg = twopi * tw * (i + 1.0) / self.w4
         j = i + self.n1 + self.n2 + self.n3
         sm += self.al[k][j] * math.cos(arg)
         sm += self.bl[k][j] * math.sin(arg) 
      return sm   
      
   # weeks - period, first - first index, nhm - number of harmonics
   def cFourier(self, weeks, first, nhm):
      swk = 7.0 * 24.0 * 3600.0 # average seconds per week 7 * 24 * 3600
      syr = swk * weeks # total seconds for a period
      mydate = self.df['Dates']
      #rdate = rdate[::-1]
      for i in range(0, self.tcrimes): # loop over all crime incidents
         t = (mydate[i]-self.t0).total_seconds()
         wks = (t / syr) % 1 # fractional part of the period
         crm = self.crime_category[i] 
         k = self.cr_a_index[crm]
         for j in range(0,nhm): # loop over Fourier components
            dj = j + 1.0
            arg = twopi * wks * dj
            cost = math.cos(arg)
            sint = math.sin(arg)
            self.an[j+first] += cost
            self.bn[j+first] += sint
            self.al[k][j+first] += cost
            self.bl[k][j+first] += sint
         #print "crime = %s are %d of %d crimes, years = %f" % (crm, freq[crm], tcrimes, wks)
      tcr = self.tcrimes + 0.0
      tcr *= 0.5
      #print "period %f weeks" % weeks
      for i in range(0, nhm):
         self.an[i+first] /= tcr
         self.bn[i+first] /= tcr
         for j in range(0, self.Nc):
            crm = self.cr_index[j]
            tla = self.freq[crm] + 0.0 
            tla *= 0.5
            self.al[j][i+first] /= tla
            self.bl[j][i+first] /= tla

