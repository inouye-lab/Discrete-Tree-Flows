
import numpy as np
import sys
sys.path.insert(1, '../DTF/')
import DTF

def counter(X,max_k,lmbda=None):
    '''
    returns a 2D array of dim-counts values 
    '''
    sums = np.sum(X.reshape((*X.shape, 1)) == np.arange(max_k), axis=0).astype('float')
    if(lmbda== None):
        return sums
    else:
        pseudo_counts = np.arange(0, lmbda*max_k, lmbda)
        return sums + pseudo_counts
        #add phesudo counts 

def count_params(T):

    current_leaves = [T.root]
    num_tree_params = 0
    num_nodes = 0
    
    while(len(current_leaves)!=0):
        #process the leaves at this level
        next_leaves = []

        for ind,current_node in enumerate(current_leaves):
            num_nodes+=1
            perm_mat = current_node.perm_matrix.copy()
            d,max_k = perm_mat.shape
            # fill in the nans with identity
            mask = np.isnan(perm_mat)
            idx, val = np.where(mask)
            perm_mat[mask] = val
            # count the identity
            identity = np.arange(max_k)
            identity_count = (perm_mat == identity).all(-1).sum()
            #print(perm_mat)
            
            non_identity_count = d - identity_count
            #print(non_identity_count)
            #print("Non identity *k")
            #print(non_identity_count*max_k)
            num_tree_params += non_identity_count*max_k
            
            if(current_node.left!=None and current_node.right!=None):
                next_leaves.append(current_node.left)
                next_leaves.append(current_node.right)
        
        current_leaves = next_leaves
    
    num_tree_params += num_nodes*2 #for split_feature and split_value in each node
    
    return num_tree_params, num_nodes
            
def update_global_counts(T,X,max_k,difference):
  
    DEBUG_update_global = False

    d = X.shape[1]
    cat_counter_all = {}
    local_counts = counter(X,max_k)
    temp_new_global_counts = np.add(difference,local_counts)
    temp_new_global_counts[np.isnan(temp_new_global_counts)] = T.global_counts[np.isnan(temp_new_global_counts)] 
    #fill in the nan values with actual values from previous global count matrix
    T.global_counts =  temp_new_global_counts

    if(DEBUG_update_global):
        print("UG:current local counts are ")
        print(local_counts)
        print("UG:difference (previous global - local) is ")
        print(difference)
        print("UG:current updated global counts are (difference plus current local)")
        print(T.global_counts)
   
def compute_prob_dist(X,perm_mat,max_k,alpha=0.01):
  '''
  Function for estimating Q based on counting the number of samples with value = 0,1,..k-1
  in X and using alpha for smoothing
  '''
  max_domain_length = np.count_nonzero(~np.isnan(perm_mat),axis = 1) #length of domain
  counts_mat = np.sum(X.reshape((*X.shape, 1)) == np.arange(max_k), axis=0).astype('float')
  counts_mat[np.isnan(perm_mat)] = np.nan
  sums = np.nansum(counts_mat,axis = 1) #sum counts, and ignore nans
  smoothed_sums = sums + alpha*max_domain_length
  return (counts_mat + alpha)  /  smoothed_sums[:,None]
  

def mat_entropy(prob):
    entropy = np.sum(np.nansum(prob * -np.log(prob),axis = 1))
    return entropy  

def calculate_jsd_sum(current_node, data_left, data_right, n_all_data, max_k):
    
    n = len(data_left) + len(data_right)
    p_data_left = len(data_left) / n
    p_data_right = len(data_right) / n
    
    #elif(weighting_strategy == 1):
    #  n = len(data_left) + len(data_right)
    #  p_data_left = (len(data_left)*n_all_data) / n
    #  p_data_right = (len(data_right)*n_all_data) / n
    #elif(weighting_strategy == 2):
    #  p_data_left = len(data_left)
    #  p_data_right = len(data_right)
    #else:
    #  p_data_left = len(data_left)/n_all_data
    #  p_data_right = len(data_right)/n_all_data
    
    q_left = compute_prob_dist(data_left,current_node.perm_matrix,max_k,alpha=0.01)
    q_right = compute_prob_dist(data_right,current_node.perm_matrix,max_k,alpha=0.01)
    q_mix = p_data_left*q_left + p_data_right*q_right
    
    return mat_entropy(q_mix) - p_data_left*mat_entropy(q_left) - p_data_right*mat_entropy(q_right)          

def calculate_wasserstein_dist(current_node,current_data, data_left, data_right, max_k):
    
    n = len(data_left) + len(data_right)
    p_data_left = len(data_left) / n
    p_data_right = len(data_right) / n
    
    q_left = compute_prob_dist(data_left,current_node.perm_matrix,max_k,alpha=0.01)
    q_right = compute_prob_dist(data_right,current_node.perm_matrix,max_k,alpha=0.01)
    q_parent = compute_prob_dist(current_data,current_node.perm_matrix,max_k,alpha=0.01)
    
    #wasserstein dist for left
    q_diff_left_parent = np.abs(q_parent - q_left)
    q_diff_left_parent_d = 0.5*np.nansum(q_diff_left_parent,axis = 1)
    w_left_parent = np.mean(q_diff_left_parent_d)
    
    #wasserstein dist for right
    q_diff_right_parent = np.abs(q_parent - q_right)
    q_diff_right_parent_d = 0.5*np.nansum(q_diff_right_parent,axis = 1)    
    w_right_parent = np.mean(q_diff_right_parent_d)
    
    return p_data_left*w_left_parent + p_data_right*w_right_parent

    

def test_tree(T,test_data):
    '''
    Given the constructed tree and test data, apply the permutations to test data
    '''
    
    DEBUG_test = False
    n_test,_ = test_data.shape
    current_node = T.root
    current_leaves = [current_node]
    current_datas = [test_data]
    current_indices = [np.array(list(range(0,n_test)))]
    new_data = test_data.copy()
    current_level = 0
    while(len(current_leaves)!=0):
        next_leaves = []
        next_indices = []
        next_data = []
        for ind,current_node in enumerate(current_leaves):
            split_feature = current_node.split_feature
            split_value = current_node.split_value
            current_data = current_datas[ind]
            indices = current_indices[ind]
            if(current_node.perm_matrix_calculated==True):
                new_data[indices,:] = np.take_along_axis(current_node.perm_matrix.T, current_data, axis = 0)   

            if(current_node.left!=None and current_node.right!=None):
                current_data = new_data[indices,:]
                data_left, indices_left, data_right, indices_right = DTF.split_data(current_data,indices.astype(int),split_feature, split_value)
            
                next_data.append(data_left)
                next_data.append(data_right)
                next_indices.append(indices_left)
                next_indices.append(indices_right)
                next_leaves.append(current_node.left)
                next_leaves.append(current_node.right)

        current_leaves = next_leaves
        current_indices = next_indices
        current_datas = next_data 
        current_level += 1

    return new_data
  
    

    
def calculate_inverse_new(T,perm_data,orig_data):
  '''
  Given the constructed tree and permuted data that has pass through that tree,
  apply the inverse permutations to the permuted data to get the data in the real domain
  '''
  DEBUG_calc_inverse = False
  n_perm,_ = perm_data.shape
  current_node = T.root
  current_leaves = [current_node]
  current_datas = [perm_data]
  current_indices = [list(range(0,n_perm))]
  new_data = []
  reverse_list =[]
  current_leve = 0
  while(current_level < max_depth):
    #process the leaves at this level
    next_leaves = []
    next_indices = []
    next_data = []
    for ind,current_node in enumerate(current_leaves):
        split_feature = current_node.split_feature
        split_value = current_node.split_value
        current_data = current_datas[ind]
        indices = current_indices[ind]
        data_left, indices_left, data_right, indices_right = split_data(current_data,indices,split_feature, split_value)
        next_data.append(data_left)
        next_data.append(data_right)
        next_indices.append(indices_left)
        next_indices.append(indices_right)
  

  
def calculate_inverse(T,perm_data,orig_data):
  '''
  Given the constructed tree and permuted data that has pass through that tree,
  apply the inverse permutations to the permuted data to get the data in the real domain
  '''
  DEBUG_calc_inverse = False
  n_perm,_ = perm_data.shape
  new_data = []
  
  for i in range(0,n_perm):
    data_point = perm_data[i,:].tolist().copy() #one sample (size d)
    

    if(DEBUG_calc_inverse):
      print("*******For data point number: "+str(i))
      orig_data_point = orig_data[i,:].tolist()
    current_node = T.root
    reverse_list = [] #to save the perm matrices in 
    
    # begin traversing the tree
    while(current_node!=None):
      if(DEBUG_calc_inverse):
        
        print("current node id is "+str(current_node.id))
        print("current node split value is "+str(current_node.split_value))

      if(current_node.perm_matrix_calculated == True):
        reverse_list.append(current_node.perm_matrix)
      
      current_split_feature = current_node.split_feature
      current_split_value = current_node.split_value
      
      if(current_split_feature!=None):
        # used any instead of ==
        #print(type(current_split_value))
        #print(type(data_point[current_split_feature]))
        #print(current_split_feature)
        #print(current_split_value)
        #print(isinstance(current_split_value, int))
        if(isinstance(current_split_value, int) and data_point[current_split_feature] == current_split_value):
          #print("left int")
          current_node = current_node.left
        elif(not isinstance(current_split_value, int) and any(data_point[current_split_feature] == current_split_value)):
          #print("left list")
          current_node = current_node.left
        else:
          #print("right")
          current_node = current_node.right
      else:
        break
    # end while loop on traversing the tree
    
    # traverse reverse list from back
    if(len(reverse_list) > 0):
      for j in range(-1,-(len(reverse_list))-1,-1):
        perm_matrix = reverse_list[j] #perm matrix of size dxk
        #print(perm_matrix.shape)
        d_perm,k_perm = perm_matrix.shape
        for i in range(0,d_perm):
            current_val = data_point[i] #the cat value of the datapoint at the i-th feature 
            perm_values = perm_matrix[i,:] #the perm values for the ith feature
            data_point[i] = np.where(perm_values==current_val)[0][0] #the index is the new value
            # because we arrange them from 0 to k          
    new_data.append(data_point)
  return np.array(new_data)
  
