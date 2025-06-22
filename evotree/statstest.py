import logging
import pandas as pd
import numpy as np
import itertools

class pps:
    def __init__(self,data=None,usedata=[]):
        """
        Posterior probability of superiority
        """
        self.data = pd.read_csv(data,header=0,index_col=0,sep='\t')
        self.usedata = usedata

    def multipaircomparison(self,fdr_threshold=0.05):
        pairwise_probs = []
        for p1,p2 in itertools.permutations(self.usedata, 2):
            prob = np.mean(self.data.loc[:,p1] > self.data.loc[:,p2])
            if prob > 0.5: pairwise_probs.append((p1, p2, prob))
        # sort by posterior probability descending (strongest evidence first)
        sorted_probs = sorted(pairwise_probs, key=lambda x: x[2], reverse=True)
        selected = [] # list of (i, j, posterior_prob)
        total_fdr = 0
        for n, (i, j, p) in enumerate(sorted_probs, start=1):
            selected.append((i, j, p))
            total_fdr = np.sum([1 - x[2] for x in selected]) / n
            if total_fdr >= fdr_threshold:
                selected.pop()
                break
        if len(selected) == 0: print("No pair is significant")
        else:
            print("Posterior Expected False Discovery Rate (PE-FDR) < {}".format(fdr_threshold))
            for i, j, prob in selected: print("{} is significantly larger than {} (PP: {})".format(i,j,prob)) 

 
