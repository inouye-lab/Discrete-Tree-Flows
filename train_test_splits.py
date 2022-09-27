
def kfold_splitter(n,k,fold_num,train_proportion):
  if k!= 1: #for kfold = 1 just use the train_proportion to split
    fold_size = n//k
    test_inds = list(range(fold_num*fold_size,(fold_num+1)*fold_size))
    set1 = set(test_inds)
    set2 = set(list(range(0,n)))
    train_inds = list(set2 - set1)
  else:
    train_inds = list(range(0,round(n*train_proportion)))
    test_inds = list(range(round(n*train_proportion),n))
  return train_inds, test_inds

  
def create_X_train_test(X,train_portion,kfolds,fold_num):
  '''
  If number of folds is 1, you need to specify the train_portion
  '''
  n,_ = X.shape
 
  train_ind_perm, test_ind_perm = kfold_splitter(n,kfolds,fold_num,train_portion)

  X_train = X[train_ind_perm,:]
  X_test = X[test_ind_perm,:]

  return X_train,X_test  