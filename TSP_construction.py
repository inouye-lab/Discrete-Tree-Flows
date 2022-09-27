import numpy as np
import time
from .helper_functions import *
from .TSP_structure import *
import sys
sys.path.insert(1, '../utilities/')
import utilities

np.random.seed(123)
#np.random.seed(42)

def traverse_tree(T):
    # traverse the tree in the order in which nodes where created
    for node in T.order:
        print("Node "+str(node.id))
        print("Node depth is "+str(node.depth))
        print("Node is leaf "+str(node.is_leaf))
        print("Length of the node samples "+str(len(node.samples)))
        print("JSD sum of the node is "+str(node.jsd_sum))
        print("split feature for the node's children is "+str(node.split_feature))
        print("split value for the node's children (left) is "+str(node.split_value))
        print("Node's perm Matrix "+str(node.perm_matrix))
  
def split_data(current_data,indices,split_feature, split_value):
        
    split_column_values = current_data[:, split_feature]
    left = np.isin(split_column_values,split_value)
    indices_left_temp = np.where(left == True)[0].flatten().astype(int)
    indices_left = indices[indices_left_temp]
    indices_right_temp = np.where(left == False)[0].flatten().astype(int)
    indices_right = indices[indices_right_temp]
    data_left = current_data[indices_left_temp,:]
    data_right = current_data[indices_right_temp,:]
        
    return data_left, indices_left, data_right, indices_right
    
def simple_split_data(current_data, split_feature, split_value):
        
    split_column_values = current_data[:, split_feature]
    left = np.isin(split_column_values,split_value)
    indices_left_temp = np.where(left == True)[0].flatten().astype(int)
    indices_right_temp = np.where(left == False)[0].flatten().astype(int)
    data_left = current_data[indices_left_temp,:]
    data_right = current_data[indices_right_temp,:]
        
    return data_left,  data_right

def get_all_possible_splits(data):
    '''
    the potential splits are for the case of categorical data, 
    all possible categories within a split_feature (column of data)
    '''
    potential_splits = {}
    _, d = data.shape
    for column_index in range(d):
        values = data[:, column_index]
        unique_values = np.unique(values)
        if len(unique_values) > 1: #if a coloumn has the same values all over, then it's useless for us to consider it as a split feature
            potential_splits[column_index] = unique_values  
    return potential_splits
   
def determine_random_split_multi(current_node,max_k,data,indices, potential_splits, split_type,SAMPLE_AT_NODE):
   
    current_node.split_feature = np.random.choice(list(potential_splits.keys()),1,replace = False)[0]
    potential_split_values = potential_splits[current_node.split_feature]
    len_cat = len(potential_split_values)
    current_node.split_value = np.random.choice(potential_split_values,len_cat//2,replace = False)

def determine_random_split(current_node,max_k,data,indices, potential_splits, split_type,SAMPLE_AT_NODE):
   
    current_node.split_feature = np.random.choice(list(potential_splits.keys()),1,replace = False)[0]
    potential_split_values = potential_splits[current_node.split_feature]
    current_node.split_value = np.random.choice(potential_split_values,1,replace = False)



def calculate_local_perm(current_node,current_data,max_k,split_feature,split_value,left_vs_right):
    #deduce the domain from the perm matrix
    DEBUG_local_perm = False
    
    domain_mask = np.isnan(current_node.perm_matrix)
    n,d = current_data.shape
    #print("domain_mask ")
    #print(domain_mask)        
    local_counts = counter(current_data,max_k)
    local_counts = local_counts.astype(float)
    local_counts[domain_mask] = np.nan 
                      
    global_rank = [list(range(0,max_k))]*d
    global_rank = np.array(global_rank).astype(float)
    global_rank[domain_mask] = np.nan
    
    if(left_vs_right=="left"): #fix here 
        #everything that is not split value on that access is nan 
        ks_range = np.array(range(0,max_k)).flatten()
        nans_indices = ks_range[np.isin(ks_range,split_value)==False]
        local_counts[split_feature, nans_indices] = np.nan
        global_rank[split_feature, nans_indices] = np.nan
    
    else:

        local_counts[split_feature, split_value] = np.nan
        global_rank[split_feature, split_value] = np.nan
    if(DEBUG_local_perm): 
        print(left_vs_right)
        print("local counts")
        print(local_counts)
    sorted_global_rank = np.argsort(global_rank, axis = 1)
    sorted_indices = np.argsort(local_counts, axis =1)
    
    temp_perm_matrix = np.empty_like(current_node.perm_matrix)
    np.put_along_axis(temp_perm_matrix,sorted_global_rank,sorted_indices,axis =1)
    
    temp_perm_matrix = temp_perm_matrix.astype(int)
    #temp_perm_matrix[domain_mask] = np.nan

    new_local_counts = np.take_along_axis(local_counts, temp_perm_matrix, axis = 1).astype(float)
    if(DEBUG_local_perm): 
        print("new_local_counts")
        print(new_local_counts)  
    return new_local_counts
    
def determine_best_split_local_perm_method_2(global_counts,current_node,max_k,data,indices, potential_splits, split_type,domain_mat,SAMPLE_AT_NODE):
    DEBUG_local_GREEDY = False
    n,_ = data.shape
    
    max_sampling_size = 1000
    current_data = data[indices,:].copy()
    n_current,_ = current_data.shape
    
    if(SAMPLE_AT_NODE):      
        sampling_size = min(n_current,max_sampling_size)
        sampled_indices = np.random.choice(n_current, sampling_size, replace=False)
        current_data = current_data[sampled_indices,:]
        n_current,_ = current_data.shape
    

    local_counts = counter(current_data,max_k)
    diff_counts = global_counts - local_counts

    best_nll = np.inf
    
    for split_feature in potential_splits: #this addresses which dimension to split on
      for value in potential_splits[split_feature]: #this addresses which value will be on the left
        
        local_data_left, local_data_right = simple_split_data(current_data, split_feature, value)
            
        new_local_counts_left = calculate_local_perm(current_node,local_data_left,max_k,split_feature,value,'left') 
        new_local_counts_right = calculate_local_perm(current_node,local_data_right,max_k,split_feature,value,'right')        
        #end_time = time.time()
        
        if(DEBUG_local_GREEDY):
            print("For split_feature "+str(split_feature)+" and split value "+str(value))
            print("Count mat left before is")
            print(counter(local_data_left,max_k))
            print("Count mat right before is")
            print(counter(local_data_right,max_k))
            
            print("Count mat left is")
            print(new_local_counts_left)
            print("Count mat right is")
            print(new_local_counts_right)
          
  
        # sums new_local_counts_right and left treating nans as zeros
        new_local_counts = np.nansum(np.dstack((new_local_counts_left,new_local_counts_right)),2)
        new_counts = diff_counts + new_local_counts
        #start_time = time.time()
        Q_temp = utilities.compute_Q_from_counts(new_counts,domain_mat,max_k,alpha=0.01)
        nll_temp = utilities.compute_avg_nll_all_Q_from_counts(new_counts,Q_temp,n)
       
        
        if(nll_temp < best_nll):
          best_nll = nll_temp
          best_split_feature = split_feature
          best_split_value = value
        
    current_node.split_feature = int(best_split_feature)
    current_node.split_value = int(best_split_value)
    

    
def fix_tree_pass3(root,d,max_k,max_depth):
    
    DEBUG_pass_3 = True
    current_node = root

    # Apply perm at root to data at root, and to split features
    #new_data[current_node.samples,:] = np.take_along_axis(current_node.perm_matrix.T, new_data[current_node.samples,:], axis = 0)
    
    # The only perm matrix that won't change is the root's perm matrix, copy it as is
    if(isinstance(current_node.split_value, np.ndarray)):
        current_node.split_value = current_node.perm_matrix[current_node.split_feature,current_node.split_value].astype(int)
    else:
        current_node.split_value = int(current_node.perm_matrix[current_node.split_feature,current_node.split_value])
    
    # save the perm at the root, because we will need it when we process children
    ancess_perms = [current_node.perm_matrix]
    
    # start fixing at level 1 of the tree since root perm is fixed by default and we have
    # already fixed the split feature at the root too
    current_leaves = [current_node.left,current_node.right]
    current_level = 1
    apply_to_all = True
    
    while(current_level <= max_depth and len(current_leaves)!=0):
        next_leaves = []
        next_ancess_perms = []
        
        #print("for level "+str(current_level))
        
        for i,current_node in enumerate(current_leaves):
            
            indices = current_node.samples
            current_depth = current_node.depth
            anc_perm_matrix = ancess_perms[i//2]

            perm_mat = current_node.perm_matrix.copy()

            #Fill the nans of the permutation matrix with identity
            mask = np.isnan(perm_mat)
            idx1, val1 = np.where(mask)
            perm_mat[mask] = val1
            #print("permutation matrix after filling in nans")
            #print(perm_mat)
            
            #Do the same for the ancestor
            mask_anc = np.isnan(anc_perm_matrix)
            idx_anc, val_anc = np.where(mask_anc)
            p_anc_all = anc_perm_matrix.copy()
            p_anc_all[mask_anc] = val_anc
          
            #permute the permutation values 
            new_permuted_matrix_vals = np.take_along_axis(anc_perm_matrix, perm_mat.astype(int), axis=1).astype(float)
            new_permuted_matrix_vals_masked = new_permuted_matrix_vals.copy()
            new_permuted_matrix_vals_masked[mask] = np.nan


            #permute the permutation indices             
            indices_matrix = np.tile(np.array(range(0,max_k)),(d,1))
            permuted_indices_matrix = np.take_along_axis(p_anc_all,indices_matrix, axis = 1)
            
            #combine the permuted indices and permuted entires to get the new permutation matrix
            np.put_along_axis(current_node.perm_matrix,permuted_indices_matrix.astype(int),new_permuted_matrix_vals_masked,axis = 1)
            
            # Update the split value of the node I am processing if it is not a leaf
            if(current_node.is_leaf == False):
                temp_split_value = anc_perm_matrix[current_node.split_feature,current_node.split_value].astype(int)
                #print("temp_split_value is "+str(temp_split_value))
                #removed ints from here
                new_split_value = current_node.perm_matrix[current_node.split_feature,temp_split_value]
                if(isinstance(new_split_value, np.ndarray)):
                    current_node.split_value = new_split_value.astype(int)
                else:
                    current_node.split_value = int(new_split_value)
                #print("old split feature "+str(temp_split_value))
                #print("new split feature "+str(current_node.split_value))
            
            if(current_node.left!=None and current_node.right!=None):
                next_ancess_perms.append(new_permuted_matrix_vals)
                next_leaves.append(current_node.left)
                next_leaves.append(current_node.right)
        current_leaves = next_leaves 
        ancess_perms = next_ancess_perms
        current_level += 1


     
      
def determine_and_apply_best_perm(current_node,data,max_k):
    '''
    determine and apply best permutations, this is done in pass 2
    '''
    DEBUG_pass_2 = False
    
    #deduce the domain from the perm matrix
    domain_mask = np.isnan(current_node.perm_matrix)
    n,d = data.shape
    
    if(DEBUG_pass_2):
        print("Calculating perm for node with ID "+str(current_node.id))
    
    if(current_node.is_leaf==True):
        
        current_data = data[current_node.samples,:]
        local_counts =  counter(current_data,max_k)
        local_counts = local_counts.astype(float)
        local_counts[domain_mask] = np.nan 
        
        global_rank = [list(range(0,max_k))]*d
        global_rank = np.array(global_rank).astype(float)
        global_rank[domain_mask] = np.nan
        
        if(DEBUG_pass_2):
            print("Global rank")
            print(global_rank)
        
        sorted_global_rank = np.argsort(global_rank, axis = 1)
        sorted_indices = np.argsort(local_counts, axis =1)
        
        if(DEBUG_pass_2):
            print("sorted_global_rank")
            print(sorted_global_rank)
            print("sorted local")
            print(sorted_indices)
            print("local counts")
            print(local_counts)
        
        np.put_along_axis(current_node.perm_matrix,sorted_indices,sorted_global_rank,axis =1)
        current_node.perm_matrix = current_node.perm_matrix.astype(float)
        current_node.perm_matrix[domain_mask] = np.nan
        
        
        #np.put_along_axis(current_node.inv_perm_matrix,sorted_global_rank,sorted_indices,axis =1)
        #current_node.inv_perm_matrix = current_node.inv_perm_matrix.astype(float)
        #current_node.inv_perm_matrix[domain_mask] = np.nan

        
        current_node.perm_matrix_calculated = True
        
        data[current_node.samples,:] = np.take_along_axis(current_node.perm_matrix.T, current_data, axis = 0)
        
        if(DEBUG_pass_2):
            local_counts =  counter(data[current_node.samples,:],max_k)
            local_counts = local_counts.astype(float)
            local_counts[domain_mask] = np.nan
            print("local counts after perms")
            print(local_counts)
    else:
        current_node.perm_matrix_calculated = True
        #for the non leafs, we only need to repermute the split features!

        split_feature = current_node.split_feature 
        # do the same just for the split feature

        current_data = data[current_node.samples,split_feature]
        feature_domain = domain_mask[split_feature,:]
        
        local_counts =  counter(current_data,max_k)
        local_counts = local_counts.astype(float)
        local_counts[feature_domain] = np.nan 
        
        global_rank = list(range(0,max_k))
        global_rank = np.array(global_rank).astype(float)
        global_rank[feature_domain] = np.nan
        
        sorted_global_rank = np.argsort(global_rank)
        sorted_indices = np.argsort(local_counts)
    
        np.put_along_axis(current_node.perm_matrix[split_feature,:],sorted_indices,sorted_global_rank,axis = 0)
        
        data[current_node.samples,split_feature] = np.take_along_axis(current_node.perm_matrix.T[:,split_feature], current_data, axis = 0)
        
        
    
    
def determine_perm_bottom_up_pass2(T,data,max_k,max_depth,level,USE_pseudo_counts,lam=10):

    '''
    determine and apply best permutation
    '''
    new_data = data.copy()
    DEBUG_buttom_up = False
    
    n,d = data.shape
    
    if(n < 20 and DEBUG_buttom_up):
        DEBUG_small_data = True #to print the data when it is small
    else:
        DEBUG_small_data = False
    
    current_leaves = T.leaves
    
    current_level = level
    if(DEBUG_buttom_up):
        print("We have "+str(len(current_leaves))+" leaves")
    
    while(current_level > 0):
        next_leaves = []
        while(len(current_leaves)!=0):
    
            current_leaf = current_leaves[0]
            current_depth = current_leaf.depth
            if(DEBUG_buttom_up):
                print("Current level is "+str(current_level))
                print("Current depth is "+str(current_depth))
                print("length of current leaves is")
                print(len(current_leaves))
            
            if(current_depth == current_level):
                left_leaf = current_leaves.pop(0)
                right_leaf = current_leaves.pop(0)
                parent_left = left_leaf.parent
                parent_right = right_leaf.parent
                
                if(parent_left == parent_right):
                    parent = parent_left
                else:
                    print("ERROR, parents don't match!!!!!!!!")
                
                current_depth = left_leaf.depth
                current_depth_right = right_leaf.depth
                
                assert current_depth == current_depth_right
                
                
                # for left data
                determine_and_apply_best_perm(left_leaf,new_data,max_k)
                
                #do the same for the right leaf
                determine_and_apply_best_perm(right_leaf,new_data,max_k)
                

                next_leaves.append(parent)
                
                if(DEBUG_buttom_up):
                    print("BU:perm matrix for left")
                    print(left_leaf.perm_matrix)
                    print("BU:perm matrix for right")
                    print(right_leaf.perm_matrix)
            else:
                
                next_leaves.append(current_leaves.pop(0))
        
        current_leaves = next_leaves
        current_level = current_level - 1
        
    #process the root
    assert len(current_leaves)==1
    
    if(USE_pseudo_counts):
        determine_and_apply_best_perm_use_pseudo_counts(current_leaves[0],new_data,max_k,lam)
    else:        
        determine_and_apply_best_perm(current_leaves[0],new_data,max_k)
    
    if(DEBUG_buttom_up):
        print("BU:perm matrix for root")
        print(current_leaves[0].perm_matrix)

    
    
def construct_tree(data,args):
    #split_type = "wasserstein_dist"
    #split_type = "JSD"
    DEBUG_construct_tree = False
    global_counts = counter(data,args.max_k)
    
    n,d = data.shape
    if(args.SAMPLE_AT_BEGIN):
        
        sampled_indices = np.random.choice(n, n//10, replace=False)
        sampled_data = data[sampled_indices,:]
        sampled_indices_list = [np.array(list(range(0,n//10)))]
    current_id = 0 #id of root is 0
       
    # Construct the tree
    T = tree_perm(n,d,args.max_k)
    
    # put nans in the appropriate place of the perm matrix at root, copy it from domain_mat
    T.root.perm_matrix[np.isnan(args.domain_mat)] = np.nan
    
    # current depth is the depth of the root at first which is 0
    current_depth = T.root.depth

    if(DEBUG_construct_tree):
      print("We now have "+str(len(T.leaves))+" leaves")
     
    # PASS 1: BFS tree construction in paper
    start_time = time.time()
    current_leaves = [T.root]
    current_level = 0

    while(current_level < args.current_max_depth):
        #process the leaves at this level
        next_leaves = []
        next_indices_list = []
        if(DEBUG_construct_tree):
            print("for level "+str(current_level))
        
        for ind,current_node in enumerate(current_leaves):
            
            indices = current_node.samples
            if(args.SAMPLE_AT_BEGIN):
                sampled_indices = sampled_indices_list[ind]
            n_current = len(indices)
            current_depth = current_node.depth

            if(DEBUG_construct_tree):
                print("+_+_+_+_+processing leaf  with id "+str(current_node.id)+" and depth is "+str(current_node.depth)+" len of samples is "+str(len(current_node.samples)))
            
            if(args.SAMPLE_AT_BEGIN):
                
                potential_splits = get_all_possible_splits(sampled_data[sampled_indices,:]) 
            else:
                potential_splits = get_all_possible_splits(data[indices,:]) 

            # start if-else split calculation 
                   
            if(current_depth < args.current_max_depth and len(potential_splits) > 0 and n_current > args.min_samples_split):
                if(args.split_type=="single_random" ):
                    
                    determine_random_split(current_node,args.max_k,data,indices, potential_splits, args.split_type,args.SAMPLE_AT_NODE)
                    _ ,indices_left ,_ ,indices_right = split_data(data[current_node.samples], current_node.samples, current_node.split_feature, current_node.split_value)   

                   
                elif(args.split_type=="greedy_local_perm"):
                    
                    #determine_best_split_local_perm_method(current_node,args.max_k,data,indices, potential_splits, args.split_type,args.domain_mat,args.SAMPLE_AT_NODE)
                    determine_best_split_local_perm_method_2(global_counts,current_node,args.max_k,data,indices, potential_splits, args.split_type,args.domain_mat,args.SAMPLE_AT_NODE)  
                    _ ,indices_left ,_ ,indices_right = split_data(data[current_node.samples], current_node.samples, current_node.split_feature, current_node.split_value)   

                T.order.append(current_node) # add this node to the order, 
                #I am using this for easy try traversal
                current_node.is_leaf = False
                
                current_id += 1
                current_node.left = tree_node(np.array(indices_left),current_id,current_node.depth+1,d,args.max_k)
                current_node.left.perm_matrix = current_node.perm_matrix.copy()
                current_node.left.parent = current_node
                
                #everything that is not split value on that access is nan 
                ks_range = np.array(range(0,args.max_k)).flatten()
                #nans_indices = ks_range[ks_range!= current_node.split_value]
                nans_indices = ks_range[np.isin(ks_range,current_node.split_value)==False]
                current_node.left.perm_matrix[current_node.split_feature, nans_indices] = np.nan
                
                
                # add the new left node as leave
                T.leaves.append(current_node.left)
                next_leaves.append(current_node.left)
                
                current_id += 1 
                current_node.right = tree_node(np.array(indices_right),current_id,current_node.depth+1,d,args.max_k) 
                current_node.right.perm_matrix = current_node.perm_matrix.copy()
                current_node.right.perm_matrix[current_node.split_feature, current_node.split_value] = np.nan
                current_node.right.parent = current_node
                
                # add the new right node as leaves
                T.leaves.append(current_node.right)
                next_leaves.append(current_node.right)  
            elif(len(potential_splits) <= 0 or n_current <= args.min_samples_split): 
                # if we have less than minimum samples, we don't split on that node
                # to do so, I set the jsd sum to a very small value
                current_node.split_feature = -1
                current_node.split_value = -1
                current_node.jsd_sum = -10**6
                next_indices_list.append([0]) #just a place holder
                next_leaves.append(current_node)
        
        if(DEBUG_construct_tree):        
            print("Done")
            print("length of next leaves is "+str(len(next_leaves)))
        
        if(len(next_leaves)!=0):        
            current_leaves = next_leaves
            sampled_indices_list = next_indices_list 
            current_level = current_level+1
        else:
            break
    T.leaves = current_leaves    
    end_time = time.time()
    pass_1_time = end_time - start_time
    
    T.order = T.order + T.leaves
    utilities.write_to_report(args.output_filename,"The depth of the tree is "+str(current_depth)+"\n")
    print("Done with Tree Construction! -- Pass 1")
    line = "Pass 1 took "+str(pass_1_time)+" seconds"
    print(line)
    utilities.write_to_report(args.output_filename,line+"\n")        
    print("We have "+str(len(T.leaves))+" leaves")
   
    ## PASS 2 -- pass1 in paper
    print("Starting perm Calculation -- Pass 2")
    start_time = time.time()
    determine_perm_bottom_up_pass2(T,data,args.max_k,args.current_max_depth,current_level,False)
    end_time = time.time()
    pass_2_time = end_time - start_time
    line = "Pass 2 took "+str(pass_2_time)+" seconds"
    print(line)
    utilities.write_to_report(args.output_filename,line+"\n")
    
    ## PASS 3 -- pass2 in paper
    print("Starting fixing the tree and perms -- Pass 3")
    start_time = time.time()
    fix_tree_pass3(T.root,d,args.max_k,args.current_max_depth)
    end_time = time.time()
    pass_3_time = end_time-start_time
    
    line = "Pass 3 took "+str(pass_3_time)+" seconds"
    print(line)
    utilities.write_to_report(args.output_filename,line+"\n")
    
    if(DEBUG_construct_tree):
      print("current_depth is "+str(current_depth))
    
    return T,current_id,pass_1_time+pass_2_time+pass_3_time
