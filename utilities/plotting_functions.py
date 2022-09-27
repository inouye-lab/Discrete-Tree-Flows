
import numpy as np
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

def plot_sample_digits(data,exp_num,fold_num,num_trees,max_depth,splitting_strategy,USE_pseudo_counts,SAMPLE_AT_NODE,SAMPLE_AT_BEGIN,orig=None):
    num_row = 5
    num_col = 5# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(3*num_col,4*num_row))
    n,d = data.shape
    new_d = int(np.sqrt(d))
    for i in range(25):
        img = np.reshape(data[i,:],(new_d,new_d))
        ax = axes[i//num_col, i%num_col]
        ax.matshow(img, cmap='gray')
    plt.tight_layout()
    fig.show()
    if(orig!=None):
        plt.savefig("Digits_ExpNum_"+str(exp_num)+"_fold_num_"+str(fold_num)+"orig.png")
    else:
        plt.savefig("Digits_ExpNum_"+str(exp_num)+"_fold_num_"+str(fold_num)+"_num_trees_"+str(num_trees)+"_max_depth_"+str(max_depth)+"_splitting_strategy_"+str(splitting_strategy)+"_use_pseudo_counts_"+str(USE_pseudo_counts)+"_sample_at_begin_"+str(SAMPLE_AT_BEGIN)+"_sample_at_node_"+str(SAMPLE_AT_NODE)+".png")

def plot_samples(Ts,Q_z,batch_size,k,data,fold_num,exp_num,max_depth,splitting_strategy,USE_pseudo_counts,SAMPLE_AT_NODE,SAMPLE_AT_BEGIN):
    probs = torch.from_numpy(Q_z)
    data_distribution = torch.distributions.OneHotCategorical(probs=probs)
    samples = data_distribution.sample([batch_size])
    data_samples = torch.argmax(samples, dim=-1).detach().numpy()

    for j in range(-1,-len(Ts)-1,-1):
        data_samples = DTF.calculate_inverse(Ts[j],data_samples,data_samples)         
    plot_sample_digits(data_samples,exp_num,fold_num,len(Ts),max_depth,splitting_strategy,USE_pseudo_counts,SAMPLE_AT_NODE,SAMPLE_AT_BEGIN,None)


def exp_8Gaussian_plotting(Ts,Q_z,batch_size,k,data,fold_num,max_depth):
    probs = torch.from_numpy(Q_z)
    data_distribution = torch.distributions.OneHotCategorical(probs=probs)
    samples = data_distribution.sample([batch_size])
    data_samples = torch.argmax(samples, dim=-1).detach().numpy()

    for j in range(-1,-len(Ts)-1,-1):
        data_samples = DTF.calculate_inverse(Ts[j],data_samples,data_samples)

    figsize = (12, 6)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    data_prob_table = np.histogramdd(data, bins=k)
    ax1.imshow(data_prob_table[0]/np.sum(data_prob_table[0]),
             cmap=cm.get_cmap("Blues", 6),
             origin="lower",
             extent=[0, k, 0, k],
             interpolation="nearest")
    ax1.set_title("Original Data Distribution")
    ax2.set_title("TSPs = "+str(len(Ts))+" M = "+str(max_depth))
    learned_prob_table = np.histogramdd(data_samples, bins=k)
    ax2.imshow(learned_prob_table[0]/np.sum(learned_prob_table[0]),
             cmap=cm.get_cmap("Blues", 6),
             origin="lower",
             extent=[0, k, 0, k],
             interpolation="nearest")
    fig.show()
    plt.savefig("Exp_8Gaussian_fold_num_"+str(fold_num)+"_num_trees_"+str(len(Ts))+"max_depth_"+str(max_depth)+".png")


    
    
def exp13_plotting(data,k):
    figsize = (6, 6)
    fig = fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
   
    data_prob_table = np.histogramdd(data, bins=k)
    ax.imshow(data_prob_table[0]/np.sum(data_prob_table[0]),
                 cmap=cm.get_cmap("Blues", 6),
                 origin="lower",
                 extent=[0, k, 0, k],
                 interpolation="nearest")
    
    ax.set_title("Data Distribution")
    fig.show()
   


def plot_bar_exp_results_diff_weights(exps_to_run,exp_results):
  fig, axes = plt.subplots(4,3 , figsize=(20, 30))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
  fig.suptitle('Bar plots for diff exps')
  j = 0
  for e,exp_result in enumerate(exp_results):
    if(e%3 == 0 and e!=0):
      j = j + 1
    exp_num = exps_to_run[e]
    bt_nll = []
    at_nll_weighting_strategy = []
    at_nll_weighting_strategy = []
    X = []
    Y = []
    for weighting_strategy in exp_result.keys():
      weighting_strategy_run = exp_result[weighting_strategy]
      num_ts = weighting_strategy_run.keys()
      #num_ts = exp_result.keys()
      for num_t in num_ts:
        max_depths = weighting_strategy_run[num_t].keys()
        for max_d in max_depths:
          avg_train_nll_bt,avg_train_nll,\
          avg_test_nll_bt,avg_test_nll,\
          avg_train_time,avg_test_time,\
          std_train_nll,std_test_nll,\
          std_train_time,std_test_time = weighting_strategy_run[num_t][max_d]
        
          X.append("Before Training")
          Y.append(avg_test_nll_bt)
         
          X.append('WS = '+str(weighting_strategy)+', \n T = '+str(num_t)+',M = '+str(max_d))
          Y.append(avg_test_nll)

          
            
    axes[j, e-(3*j)].set_title("Exp "+str(exp_num))
    axes[j, e-(3*j)].set_xticklabels(labels = X, rotation=45)
    axes[j, e-(3*j)].set_yscale("log")
    sns.barplot(ax=axes[j, e-(3*j)], y = Y, x=X)

def plot_exp_folds(folds,exp,nll_bt,nll_at,splitting_strategy,exp_num,max_depth,num_trees,USE_pseudo_counts,SAMPLE_AT_BEGIN,SAMPLE_AT_NODE):
  '''
  plot only for different lambdas and one max depth and one num trees
  '''

  fig,ax=plt.subplots()
  ax.set_xlabel("Fold #")
  ax.set_ylabel("NLL")
  ax.semilogy(list(range(0,folds)), nll_bt,'-ro',list(range(0,folds)), nll_at,'-bo')

  extra_str = 'Weighting strategy = '+str(splitting_strategy)+'\n'
  plt.title(extra_str + 'NLLs before and after training,\nfor exp = '+str(exp)+', max_depth = '+str(max_depth)+', num_trees = '+str(num_trees))
  plt.legend(["NLL before Training","NLL after Training"],prop={'size': 10})
  #plt.show()
  plt.tight_layout()
  plt.savefig("ExpNum_"+str(exp_num)+"_splitting_strategy_"+str(splitting_strategy)+"_use_pseudo_counts_"+str(USE_pseudo_counts)+"_sample_at_begin_"+str(SAMPLE_AT_BEGIN)+"_sample_at_node_"+str(SAMPLE_AT_NODE)+".png")

