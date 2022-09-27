import numpy as np


# Likelihood Helper Functions 
def compute_Q_from_counts(counts_mat,domain_mat,max_k,alpha=0.01):
  '''
  Function for estimating Q based on counting the number of samples with value = 0,1,..k-1
  in X and using alpha for smoothing
  '''
  max_domain_length = np.count_nonzero(~np.isnan(domain_mat),axis = 1) #length of domain
  counts_mat[np.isnan(domain_mat)] = np.nan
  sums = np.nansum(counts_mat,axis=1)
  smoothed_sums = sums + alpha*max_domain_length
  return (counts_mat + alpha)  /  smoothed_sums[:,None]


def compute_avg_nll_all_Q_from_counts(counts,Q,n):
  return np.nansum(-(counts/n)*np.log(Q))
  


def compute_Q(X,domain_mat,max_k,alpha=0.01):
  '''
  Function for estimating Q based on counting the number of samples with value = 0,1,..k-1
  in X and using alpha for smoothing
  '''
  max_domain_length = np.count_nonzero(~np.isnan(domain_mat),axis = 1) #length of domain
  counts_mat = np.sum(X.reshape((*X.shape, 1)) == np.arange(max_k), axis=0).astype('float')
  counts_mat[np.isnan(domain_mat)] = np.nan
  sums = np.nansum(counts_mat,axis=1)
  smoothed_sums = sums + alpha*max_domain_length
  return (counts_mat + alpha)  /  smoothed_sums[:,None]

  
def compute_avg_nll_all_Q(X,Q):

  n,d = X.shape
  probs = np.take_along_axis(Q.T,X,axis = 0)
  log_probs = np.log(probs)
  sum_samples_log_prob = np.sum(np.sum(log_probs,axis=1))
  avg_samples_log_prob = sum_samples_log_prob/n
  
  return -1*sum_samples_log_prob,-1*avg_samples_log_prob

