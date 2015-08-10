   def __init__(self, PDs, cr_index, na, nb, xmin, xmax, ymin, ymax):
      self.NPD = na # Number of police departments 
      self.Ncc = nb # Number of crime categories
      self.PDs = PDs 
      self.cr_index = cr_index
      self.cr_a_index = pd.DataFrame(data = np.arange(self.Ncc, dtype=np.int), index = self.cr_index) 
      self.cr_a_index = self.cr_a_index[0] # put crime catagory in, get int out
      self.prpd = np.empty([self.NPD, self.Ncc]) # probabilities for each police department
      self.Nadds = 0
      self.Naddp = 0
      self.pradds = 0
      self.pd_a_index = pd.DataFrame(data = np.arange(self.NPD, dtype=np.int), index = self.PDs)
      self.pd_a_index = self.pd_a_index[0] # PD index, put PD in, get int out
      self.addy_a_index = 0

      #self.Ngxy1 = np.zeros(Ntgl1, dtype=np.int)
      #self.prgrid1 = np.zeros([Ntgl1, self.Ncc])
      #self.Ngxy2 = np.zeros(Ntgl2, dtype=np.int)
      #self.prgrid2 = np.zeros([Ntgl2, self.Ncc])
      self.NicA = 0 # will get number of Addresses later
      self.NicG1 = np.zeros(Ntgl1, dtype=np.int)
      self.NicG2 = np.zeros(Ntgl2, dtype=np.int)
      self.NicPD = np.zeros(self.NPD, dtype=np.int)
      # Now probability densities
      self.PicA = 0
      self.PicG1 = np.zeros(Ntgl1)
      self.PicG2 = np.zeros(Ntgl2)
      self.PicPD = np.zeros(self.NPD)
   def train(self, by_pd, by_add, N_add, df1):
      # Do PD's
      for i in range (0, self.NPD):
         self.normpd(self.prpd[i], by_pd[self.PDs[i]], self.Ncc) # PD's delt with
      #########################
      # Now deal with addresses
      self.N_addp = N_add[N_add >= Ncm]
      self.Nadds = len(self.N_addp) # number of addresses where at least Ncm of a crime category has occured
      print "There are %d addresses where at least %d of a crime category has occured" % (len(self.N_addp), Amin)   
      self.NicA = np.empty([self.Nadds, self.Ncc], dtype=np.int) # raw counts for each top address
      self.PicA = np.empty([self.Nadds, self.Ncc]) # probabilities for each top address
      for i in range(0, self.Nadds):
         self.normpd(self.pradds[i], by_add[self.N_addp.index[i]], self.Ncc)
      self.addy_a_index = pd.DataFrame(data = np.arange(self.Nadds, dtype=np.int), index = self.N_addp.index.values)
      self.addy_a_index = self.addy_a_index[0]
      ######################
      # Finally do the grid
      N = len(df1)
      print "X length = %d" % N
      for i in range(N):
         if df1.Y[i] < 40.0: # work out grid #
            addy = df1.Address[i]
            try:
               self.N_addp.loc[addy]  # address is being used, so skip
            except KeyError:         
               ix = int((df1.X[i] - self.x0) / self.dx1)
               iy = int((df1.Y[i] - self.y0) / self.dy1)
               ig = iy * Ngl1 + ix  
               self.prgrid1[ig][self.cr_a_index[df1.Category[i]]] += 1.0 # grid 1
               self.Ngxy1[ig] += 1
               ix = int((df1.X[i] - self.x0) / self.dx2)
               iy = int((df1.Y[i] - self.y0) / self.dy2)
               ig = iy * Ngl2 + ix  
               self.prgrid2[ig][self.cr_a_index[df1.Category[i]]] += 1.0 # grid 2
               self.Ngxy2[ig] += 1
      for i in range(Ntgl1):
         self.normgd(self.prgrid1[i])
      for i in range(Ntgl2):
         self.normgd(self.prgrid2[i])
      n1 = 0
      n2 = 0
      for i in range(N):
         if df1.Y[i] > 40.0:
            continue
         ix = int((df1.X[i] - self.x0) / self.dx1)
         iy = int((df1.Y[i] - self.y0) / self.dy1)
         ig1 = iy * Ngl1 + ix 
         ix = int((df1.X[i] - self.x0) / self.dx2)
         iy = int((df1.Y[i] - self.y0) / self.dy2)
         ig2 = iy * Ngl2 + ix   
         if self.Ngxy1[ig1] >= Gmin:
            n1 += 1
         elif self.Ngxy2[ig2] >= Gmin:
            n2 += 1
      print "n1 = %d, n2 = %d" % (n1, n2)
   ## Training Finished
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
   def normgd(self, pr):
      tot = sum(pr)
      if tot > 0.1:
         for i in range(self.Ncc):
            pr[i] = pr[i] / tot
   def predict(self, PD, addy, X, Y):
      try:
         N = self.N_addp.loc[addy]
         i = self.addy_a_index[addy]
         return self.pradds[i]
      except KeyError: # won't use the address then
         if Y < 40.0: # work out grid #
            ix = int((X - self.x0) / self.dx1)
            iy = int((Y - self.y0) / self.dy1)
            ig = iy * Ngl1 + ix
            if self.Ngxy1[ig] >= Gmin:
               return self.prgrid1[ig]
            ix = int((X - self.x0) / self.dx2)
            iy = int((Y - self.y0) / self.dy2)
            ig = iy * Ngl2 + ix
            if self.Ngxy2[ig] >= Gmin:
               return self.prgrid2[ig]
         i = self.pd_a_index[PD]
         #print 'Oh no %d' % i
         return self.prpd[i]
         
