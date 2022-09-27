import numpy as np
from .helper_functions import *

class tree_node():
  def __init__(self,samples,id,depth,d,k):
    self.id = id #id of the tree node    
    self.samples = samples #indices for the samples related to this tree node
    self.depth = depth #the depth of the current node    
    self.parent = None #the parent of the current node
    self.left = None #pointer to the left node
    self.right = None #pointer to the right node
    self.is_leaf = True #whether the node is a leaf or not
    self.split_feature = None #which feature to flip
    self.split_value = None #this is basically split_value_left
    self.perm_matrix = np.full((d,k),list(range(0,k))).astype(float)
    self.perm_matrix_calculated = False 
    #this is not used
    self.counts_matrix = np.full((d,k),np.nan).astype(float)
    self.jsd_sum = None
    
    
 
  
class tree_perm():
  def __init__(self,n,d,k): #possibly remove domain later
    self.root = tree_node(np.array(list(range(0,n))),0,0,d,k) #samples, id, depth, dimention, max k, domain
    self.leaves = [self.root] #at the begining the root is the only leaf
    self.order = [] #pointer to the order of tree construction, helpful in test
    self.global_counts = None
    #self.global_counts = np.full((d,k),np.nan) #the global counts matrix
