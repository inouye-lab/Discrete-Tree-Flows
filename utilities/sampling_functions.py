import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import figure
import matplotlib.colors as mcolors
import seaborn as sns
import torch
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../DTF/')
import DTF
torch.manual_seed(42)


def sample_and_invert_snps(Ts,Q_z,batch_size,k,fold_num,exp_num,max_depth):
    print("Sampling from Q_z....")
    probs = torch.from_numpy(Q_z)
    data_distribution = torch.distributions.OneHotCategorical(probs=probs)
    samples = data_distribution.sample([batch_size])
    data_samples = torch.argmax(samples, dim=-1).detach().numpy()

    for j in range(-1,-len(Ts)-1,-1):
        data_samples = DTF.calculate_inverse(Ts[j],data_samples,data_samples)         
    
    ag = np.array(["AG"]*batch_size)
    ag = ag[:,np.newaxis]
    nums = np.array(list(range(0,batch_size)))
    nums = nums[:,np.newaxis]
    nums = np.char.mod('%d', nums) 
    new = np.hstack((ag,nums))
    ag_num = pd.DataFrame(new, columns = ['AG','val2'])
    ag_num["AG_num"] = ag_num["AG"].astype(str) +""+ ag_num["val2"]
    ag_num = ag_num.drop("val2",axis=1)
    data_samples = np.hstack((ag_num.values,data_samples))
    data_samples = pd.DataFrame(data_samples)
    #Output AGs in hapt format
    data_samples.to_csv("Exp_num_"+str(exp_num)+"_num_trees_"+str(len(Ts))+"_max_depth_"+str(max_depth)+"_fold_num_"+str(fold_num)+"_output.hapt", sep=" ", header=False, index=False)
