import pandas as pd
import numpy as np

Amin = 1000 # minimum counts to use address

class posd:
   def __init__(self, PDs, cr_index, na, nb):
      self.NPD = na
      self.Ncc = nb
      self.PDs = PDs
      self.cr_index = cr_index
      print "cr_index"
      print cr_index
      self.cr_a_index = pd.DataFrame(data = np.arange(self.Ncc, dtype=np.int), index = self.cr_index) 
      self.cr_a_index = self.cr_a_index[0] # put crime catagory in, get int out
      self.prpd = np.empty([self.NPD, self.Ncc]) # probabilities for each police department
      self.Nadds = 0
      self.Naddp = 0
      self.pradds = 0
      print "pd_index"
      print PDs
      self.pd_a_index = pd.DataFrame(data = np.arange(self.NPD, dtype=np.int), index = self.PDs)
      self.pd_a_index = self.pd_a_index[0]
      self.addy_a_index = 0
   def train(self, by_pd, by_add, N_add):
      for i in range (0, self.NPD):
         self.normpd(self.prpd[i], by_pd[self.PDs[i]], self.Ncc)
      self.N_addp = N_add[N_add > Amin]
      print "There are %d addresses where at least %d crimes have occured" % (len(self.N_addp), Amin)   
      #print N_add['2000 Block of MISSION ST']
      #print by_add['2000 Block of MISSION ST'].sum()
      self.Nadds = len(self.N_addp)
      self.pradds = np.empty([self.Nadds, self.Ncc]) # probabilities for each top address
      for i in range (0, self.Nadds):
         self.normpd(self.pradds[i], by_add[self.N_addp.index[i]], self.Ncc)
      #print "addy_index"
      #print self.N_addp.index.values
      self.addy_a_index = pd.DataFrame(data = np.arange(self.Nadds, dtype=np.int), index = self.N_addp.index.values)
      self.addy_a_index = self.addy_a_index[0]
   def normpd(self, pr, raw, N):
      norm = 0.0
      pr0 = np.zeros(N, dtype=np.int)
      rl = len(raw)
      for i in range (0, rl):
         j = self.cr_a_index[raw.index[i]]
         pr0[j] += raw.values[i]
      for i in range (0, N):
         pr[i] = pr0[i] + 0.0
         norm += pr[i]
      pr /= norm
   def predict(self, PD, addy):
      try:
         N = self.N_addp.loc[addy]
         i = self.addy_a_index[addy]
         #print addy
         #print 'nice :) % i' % i
         return self.pradds[i]
      except KeyError:
         i = self.pd_a_index[PD]
         #print 'Oh no %d' % i
         return self.prpd[i]
         

