import glob, os
import numpy as np
import pickle

def get_filenames(main_path,exp_num,sample_node,args):
    
    path_to_tree_results = main_path+"exp"+str(exp_num)+"/TSPs"+str(args.num_TSPs)+"/depth"+str(args.max_depth)+"/split_type_"+args.split_type+"/nfolds"+str(args.kfolds)+"/"
    
    os.chdir(path_to_tree_results)
    print("Currently in directory")
    print(os.getcwd())
    filenames_dict_nll = {}
    if(sample_node == True):
        for file in glob.glob("*_True.obj"):
            #print(file)
            max_depth_preprocess = file.split("max_depth_",1)[1]
            if(max_depth_preprocess[1].isdigit()):
                max_depth = max_depth_preprocess[0:2]
            else:
                max_depth = max_depth_preprocess[0]
            num_trees_preprocess = file.split("current_tree_",1)[1]
            if(num_trees_preprocess[2].isdigit()):
                num_trees = num_trees_preprocess[0:3]
            elif(num_trees_preprocess[1].isdigit()):
                num_trees = num_trees_preprocess[0:2]
            else:
                num_trees = num_trees_preprocess[0]
            splitting_start_preprocess = file.split("splitstrat_",1)[1]
            if(splitting_start_preprocess[0]=='g'):
                split_strat = 'greedy_local_perm'
            elif(splitting_start_preprocess[0]=='J'):
                split_strat = 'JSD'
            elif(splitting_start_preprocess[0]=='s'):
                split_strat = 'single_random'
            else: #(splitting_start_preprocess[0]=='r'):
                split_strat = 'random_multi'
            if (int(max_depth),int(num_trees),split_strat) not in filenames_dict_nll: 
                filenames_dict_nll[(int(max_depth),int(num_trees),split_strat)] = [file]
            else:
                filenames_dict_nll[(int(max_depth),int(num_trees),split_strat)].append(file)
    else:
        for file in glob.glob("*_False.obj"):
            
            max_depth_preprocess = file.split("max_depth_",1)[1]
            if(max_depth_preprocess[1].isdigit()):
                max_depth = max_depth_preprocess[0:2]
            else:
                max_depth = max_depth_preprocess[0]
            num_trees_preprocess = file.split("current_tree_",1)[1]
            if(num_trees_preprocess[2].isdigit()):
                num_trees = num_trees_preprocess[0:3]
            elif(num_trees_preprocess[1].isdigit()):
                num_trees = num_trees_preprocess[0:2]
            else:
                num_trees = num_trees_preprocess[0]
            splitting_start_preprocess = file.split("splitstrat_",1)[1]
            if(splitting_start_preprocess[0]=='g'):
                split_strat = 'greedy_local_perm'
            elif(splitting_start_preprocess[0]=='J'):
                split_strat = 'JSD'
            elif(splitting_start_preprocess[0]=='s'):
                split_strat = 'single_random'
            else: #(splitting_start_preprocess[0]=='r'):
                split_strat = 'random_multi'
            if (int(max_depth),int(num_trees),split_strat) not in filenames_dict_nll: 
                filenames_dict_nll[(int(max_depth),int(num_trees),split_strat)] = [file]
            else:
                filenames_dict_nll[(int(max_depth),int(num_trees),split_strat)].append(file)

    return filenames_dict_nll

def print_avg_fold_results(exp,sampling,best_results):
    # to help directly print on overleaf
    print("________________________________")
    print("For exp {} and sampling {}".format(exp, sampling))
    print("num trees = {}, M = {}".format(best_results["num_trees"],best_results["depth"]))
    print("train nll: {} $\pm$ {}".format(round(best_results["train"],2),round(best_results["train_std"],2)))   
    print("test nll: {} $\pm$ {}".format(round(best_results["test"],2),round(best_results["test_std"],2)))
    print("train bpc: {} $\pm$ {}".format(round(best_results["train_bpc"],2),round(best_results["train_bpc_std"],2)))   
    print("test bpc: {} $\pm$ {}".format(round(best_results["test_bpc"],2),round(best_results["test_bpc_std"],2)))
    
    print("train time: {} $\pm$ {}".format(round(best_results["train_t"],2),round(best_results["train_t_std"],2)))
    print("test time: {} $\pm$ {}".format(round(best_results["test_t"],2),round(best_results["test_t_std"],2)))  
    print("num paramters: {} $\pm$ {}".format(round(best_results["params"],2),round(best_results["params_std"],2)))

def find_fold_average(exp,filenames_dict,expected_nfolds,max_depth,num_trees,max_num_trees,split_strat,sampling):

    unsorted_keys = list(filenames_dict.keys())
    sorted_keys = sorted(unsorted_keys, key=lambda element: (element[0], element[1]))
    new_results = {}
    print(filenames_dict)
    filenames_all_folds = filenames_dict[(max_depth,max_num_trees,split_strat)]
    nfolds = len(filenames_all_folds)
    
    if(nfolds!=expected_nfolds):
        print(filenames_all_folds)
        print(expected_nfolds)
        print(nfolds)
    
    assert nfolds==expected_nfolds
    all_folds_train_nll = []
    all_folds_test_nll = []
    all_folds_val_nll = []
    all_folds_num_nodes = []
    all_folds_params = []
    all_folds_train_time = []
    all_folds_test_time = []
    all_folds_val_time = []
    all_folds_train_bpc = []
    all_folds_test_bpc = []
    for filename in filenames_all_folds:
        splitted_line = filename.split("splitstrat_",1)[1]
        
        tree_result_file = open(filename, 'rb') 
        result = pickle.load(tree_result_file)
        tree_result_file.close()
        
        #num_trees = max_num_trees
        all_folds_train_nll.append(result["train"][num_trees])
        all_folds_test_nll.append(result["test"][num_trees])
        all_folds_train_bpc.append(result["train_bpc"][num_trees])
        all_folds_test_bpc.append(result["test_bpc"][num_trees])
        #all_folds_val_nll.append(result["val"][num_trees-1])
        all_folds_num_nodes.append(result["num_nodes"][num_trees])
        all_folds_params.append(result["params"][num_trees])
        all_folds_train_time.append(result["train_t"][num_trees])
        all_folds_test_time.append(result["test_t"][num_trees])
    #all_folds_val_time.append(result["val_t"][num_trees-1])
    new_results["depth"] = max_depth
    new_results["num_trees"] = num_trees + 1
    #means across nfolds
    new_results["train"] = np.mean(all_folds_train_nll)
    new_results["test"] = np.mean(all_folds_test_nll)
    #new_results["val"] = np.mean(all_folds_val_nll)
    new_results["train_t"]  = np.mean(all_folds_train_time)
    new_results["test_t"] = np.mean(all_folds_test_time)
    #new_results["val_t"]  = np.mean(all_folds_val_time)
    new_results["num_nodes"]  = np.mean(all_folds_num_nodes)
    new_results["params"]  = np.mean(all_folds_params)
    new_results["train_bpc"] = np.mean(all_folds_train_bpc)
    new_results["test_bpc"] = np.mean(all_folds_test_bpc)
    #std across nfolds
    new_results["train_std"] = np.std(all_folds_train_nll)
    new_results["test_std"] = np.std(all_folds_test_nll)
    #new_results["val_std"] = np.std(all_folds_val_nll)
    new_results["train_t_std"]  = np.std(all_folds_train_time)
    new_results["test_t_std"] = np.std(all_folds_test_time)
    #new_results["val_t_std"]  = np.std(all_folds_val_time)
    new_results["num_nodes_std"]  = np.std(all_folds_num_nodes)
    new_results["params_std"]  = np.std(all_folds_params)
    new_results["train_bpc_std"] = np.std(all_folds_train_bpc)
    new_results["test_bpc_std"] = np.std(all_folds_test_bpc)
    
    print_avg_fold_results(exp,sampling,new_results)
        
    return new_results


'''
sample_nodes = [True]
exps = [3,4,5,6,7,8,9,10]
main_path = "/tree_results/" #specify the main path where the exp results are present

for sample_at_node in sample_nodes:
    for exp_num in exps:
        if(exp_num==11 or exp_num==14): #3folds for mnist or genetic, 5 otherwise
            expected_nfolds = 3
        else:
            expected_nfolds = 5
        filenames_dict_nll = get_filenames(main_path,exp_num,sample_at_node)
        #print(filenames_dict_nll)
        
        new_results,best_results = find_fold_averages(exp_num,filenames_dict_nll,expected_nfolds)
        
        print_best_results(exp_num,sample_at_node,best_results)
        
'''
        
