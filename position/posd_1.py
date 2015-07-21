import pandas as pd
import numpy as np

class posd:
   def __init__(self, cr_index, na, nb):
      self.NPD = na
      self.Ncc = nb
      self.cr_index = cr_index
      self.cr_a_index = pd.DataFrame(data = np.arange(self.Ncc, dtype=np.int), index = self.cr_index) 
      self.cr_a_index = self.cr_a_index[0] # put crime catagory in, get int out
   def norm(self, pr, raw):
      norm = 0.0
      pr0 = np.zeros(self.Ncc, dtype=np.int)
      rl = len(raw)
      for i in range (0, rl):
         j = self.cr_a_index[raw.index[i]]
         pr0[j] += raw.values[i]
      for i in range (0, self.Ncc):
         pr[i] = pr0[i] + 0.0
         norm += pr[i]
      pr /= norm

