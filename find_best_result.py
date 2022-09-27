import glob, os
import numpy as np
import pickle
import math
from argparse import ArgumentParser

def get_filenames(args,sample_node):
    cwd = os.getcwd()
    path_to_tree_results = cwd+"/fold_results/exp"+args.exp+"/split_type_"+args.split_type+"/nfolds"+str(args.kfolds)+"/"
    
    os.chdir(path_to_tree_results)
    filenames_dict_nll = {}
    if(sample_node == True):
        for file in glob.glob("*_True.obj"):
            #print(file)
            max_depth_preprocess = file.split("max_depth_",1)[1]
            if(max_depth_preprocess[1].isdigit()):
                max_depth = max_depth_preprocess[0:2]
            else:
                max_depth = max_depth_preprocess[0]
            num_trees_preprocess = file.split("max_num_trees_",1)[1]
            if(num_trees_preprocess[1].isdigit()):
                num_trees = num_trees_preprocess[0:2]
            else:
                num_trees = num_trees_preprocess[0]
            if (int(max_depth),int(num_trees)) not in filenames_dict_nll: 
                filenames_dict_nll[(int(max_depth),int(num_trees))] = [file]
            else:
                filenames_dict_nll[(int(max_depth),int(num_trees))].append(file)
    else:
        for file in glob.glob("*_False.obj"):
            #print(file)
            max_depth_preprocess = file.split("max_depth_",1)[1]
            if(max_depth_preprocess[1].isdigit()):
                max_depth = max_depth_preprocess[0:2]
            else:
                max_depth = max_depth_preprocess[0]
            num_trees_preprocess = file.split("max_num_trees_",1)[1]
            if(num_trees_preprocess[1].isdigit()):
                num_trees = num_trees_preprocess[0:2]
            else:
                num_trees = num_trees_preprocess[0]
            if (int(max_depth),int(num_trees)) not in filenames_dict_nll: 
                filenames_dict_nll[(int(max_depth),int(num_trees))] = [file]
            else:
                filenames_dict_nll[(int(max_depth),int(num_trees))].append(file)

    return filenames_dict_nll

def find_best_result(filenames_dict):

    best_test_nll = 10**6
    unsorted_keys = list(filenames_dict.keys())
    sorted_keys = sorted(unsorted_keys, key=lambda element: (element[0], element[1]))
    
    for max_depth, num_trees in sorted_keys:
        filename = filenames_dict[(max_depth,num_trees)][0]
        
        tree_result_file = open(filename, 'rb') 
        result = pickle.load(tree_result_file)
        tree_result_file.close()
        
        for num_tree in range(0,num_trees):
            current_result = result[num_tree]
            other_condition = (((best_test_nll - current_result["test"])/best_test_nll) > 10**-2)
            if(current_result["test"] < best_test_nll and other_condition):
                best_test_nll = current_result["test"]
                best_result = current_result
    return best_result

def print_results(new_results):
    for k,v in new_results.items():
        print("________________________________")
        print(k)
        for j in range(0,len(new_results[k])):
            r = new_results[k][j]
            print("________________________________")
            for key,val in r.items():
                print(str(key)+" : "+str(val))

def print_best_results_latex(exp,sampling,best_results):
    print("________________________________")
    print("For exp {} and sampling {}".format(exp, sampling))
    print("\\shortstack{{nTSP = {}\\\\ M = {}}}".format(best_results["num_trees"],best_results["depth"]))
    print("test nll: {} \\tiny{{($\pm$ {})}}".format(round(best_results["test"],2),round(best_results["test_std"],2)))
    print("train time: {} \\tiny{{($\pm$ {})}}".format(round(best_results["train_t"],1),round(best_results["train_t_std"],1)))
    print("train time: {} +/- {}".format(round(best_results["train_t"],1),round(best_results["train_t_std"],1)))
    print("num paramters: {} \\tiny{{($\pm$ {})}}".format(math.ceil(best_results["params"]),int(best_results["params_std"])))
    print("test time: {} $\pm$ {}".format(round(best_results["test_t"],4),round(best_results["test_t_std"],4)))  

def print_best_results(args,sample_at_node,best_results):
    print("________________________________")
    print("For exp {}, split type: {} and nfolds: {}".format(args.exp, args.split_type, args.kfolds))
    print("Best result was for:")
    print("For nTSP = {} and M = {}".format(best_results["num_trees"],best_results["depth"]))
    print("test nll: {} +/- {}".format(round(best_results["test"],2),round(best_results["test_std"],2)))
    print("train time: {} +/- {}".format(round(best_results["train_t"],2),round(best_results["train_t_std"],2)))
    print("train time: {} +/- {}".format(round(best_results["train_t"],2),round(best_results["train_t_std"],2)))
    print("num paramters: {} +/- {}".format(math.ceil(best_results["params"]),int(best_results["params_std"])))
    print("test time: {} +/- {}".format(round(best_results["test_t"],2),round(best_results["test_t_std"],2)))  
        


#example run:
#python find_best_result.py --exp=8Gaussian --kfolds=5 --split_type=single_random 
        
parser = ArgumentParser()

parser.add_argument("--exp", type=str, default="8Gaussian", help="exp name COPH, COPM, COPW, 8Gaussian, MNIST, SNP, Mushroom")
parser.add_argument("--split_type", type=str, default="single_random", help="The type of split to use, random vs GLP")
parser.add_argument("--kfolds", type=int, default=1, help="The number of kfolds to test")

args = parser.parse_args()

if(args.exp=="MNIST" or args.exp=="SNP"):
    sample_at_node = True
else:
    sample_at_node = False

filenames_dict_nll = get_filenames(args,sample_at_node)

best_result = find_best_result(filenames_dict_nll)

print_best_results(args,sample_at_node,best_result)
