import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from datetime import datetime

twopi = 2.0 * math.pi
nw = 645.143
Nmin = 1000 # twice the minimum possible number in each sub division 
bmax = 16 # Max number of bins
Dmin = 2000 # minimum category count to modulate according to day 
Hmin = 2000 # minimum category count to modulate according to hour 
#t0 = np.datetime64("2003-01-01")

class sdiv:
   def __init__(self, df, t0):    
      self.t0 = t0
      self.df = df
      group = self.df.groupby('Category')
      self.crime_category = self.df['Category'] # list of crime types
      self.freq = group.size()   # histogram of crime types
      self.cr_index = self.freq.index.values  # put int in, get crime category out
      self.tcrimes = self.freq.sum() # total number of crimes in training data
      self.Ncc = len(self.cr_index)    # number of crime categories
      self.cr_a_index = pd.DataFrame(data = np.arange(self.Ncc, dtype=np.int), index = self.cr_index) 
      self.cr_a_index = self.cr_a_index[0] # put crime catagory in, get int out
      # arrays
      self.Nsd = np.empty(self.Ncc, dtype=np.int) # Number of sub divisions, for each crime type
      self.Nsc = np.zeros((self.Ncc, bmax), dtype=np.int) # 2D array for histograms of subtotal crimes
      self.days = np.empty((self.Ncc, 7)) # 2D array for modulation according to day
      #self.Nsa = np.zeros(bmax, dtype=np.int) # array for sum of all crime categories at bmax time intervals
      self.wa = np.empty(self.Ncc) # weeks in each subdivision
      self.hrs = np.empty((self.Ncc, 24)) # 2D array for modulation according to hour of day
      #t1 = np.datetime64("2003-01-01")
      #t2 = np.datetime64("2015-05-14")
      #td = (t2-t1).item().total_seconds()
      #td = td / (7.0 * 24.0 * 3600.0)
      #print "number of weeks = %f" % td
   def train(self):
      for i in range(0, self.Ncc):
         self.Nsd[i] = self.freq[i] / Nmin
         self.Nsd[i] += 1 
         if self.Nsd[i] > bmax:
			   self.Nsd[i] = bmax
         self.wa[i] = nw / (self.Nsd[i] + 0.0) # no. weeks per bin
      mydates = self.df['Dates'] # array of dates
      myd0 = copy.deepcopy(mydates)
      myd0 = myd0.apply(lambda x: x.replace(hour=0, minute=0))
      idays = np.zeros((self.Ncc, 7), dtype=np.int) # 2D array for modulation according to day
      ihrs = np.zeros((self.Ncc, 24), dtype=np.int) # 2D array for modulation according to hour
      sday = 24.0 * 3600.0 # seconds per day
      swk = 7.0 * sday # seconds per week 7 * 24 * 3600
      shr = 3600.0 # seconds per hour
      for i in range(0, self.tcrimes): # loop over all crime incidents
         crm = self.crime_category[i] 
         j = self.cr_a_index[crm] # integer index to type of crime
         t = (mydates[i]-myd0[i]).total_seconds() # make sure hour not effected by daylight savings etc
         t /= 3600.0
         hrd = int(t)
         ihrs[j, hrd] += 1
         t = (mydates[i]-self.t0).total_seconds()
         wks = t / swk # time in weeks
         k = int(wks / self.wa[j])
         self.Nsc[j, k] += 1  #index array for category drift
         dow = int(t / sday) # day of week, clicks over at 6am
         if hrd >= 6: 
            dow += 1
         dow = dow % 7 
         idays[j, dow] += 1
      for i in range(0, self.Ncc):
         for j in range(0, 7):
            self.days[i,j] = (idays[i,j] + 0.0) * 7.0 / (self.freq[i] + 0.0)
         for j in range(0, 24):
            self.hrs[i,j] = (ihrs[i,j] + 0.0) * 24.0 / (self.freq[i] + 0.0)
         for j in range(0, bmax):
            if j < self.Nsd[i]:
               self.Nsc[i, j] = self.Nsc[i, j] * self.Nsd[i]
   def getProb(self, tw, th, pr):
      tot = 0.0
      hrd = int(th)
      dow = int(tw * 7.0)
      if hrd >= 6:
         dow += 1
      dow = dow % 7
      for i in range(0, self.Ncc):
         j = int(tw / self.wa[i])
         #self.Nsd[i] # number of sub-divisions 
         pr[i] = self.Nsc[i, j] + 0.0
      for i in range(0, self.Ncc):
         if self.freq[i] > Dmin:
            pr[i] *= self.days[i,dow]
         if self.freq[i] > Hmin:
            pr[i] *= self.hrs[i,hrd]
         tot += pr[i]
      for i in range(0, self.Ncc):
         pr[i] = pr[i] / (tot + 0.0)


