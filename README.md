# Discrete Tree Flows
This repository contains the official code for Discrete Tree Flows (DTF) from the following ICML paper.

Mai Elkady, Jim Lim, and David I. Inouye.  
Discrete tree flows via tree-structured permutations.  
In *International Conference on Machine Learning (ICML)*, 2022, 17-23 July 2022, Baltimore,
Maryland, USA.

BibTex:
```
@inproceedings{elkady2022discrete,
 author = {Mai Elkady and Jim Lim and David I. Inouye},
 booktitle = {International Conference on Machine Learning (ICML)},
 title = {Discrete Tree Flows via Tree-Structured Permutations},
 year = {2022},
}
```

# Description
Discrete flow-based models cannot be straightforwardly optimized with conventional deep learning methods because gradients of discrete functions are undefined or zero. Our approach seeks to reduce computational burden by developing a discrete flow based on decision trees—building upon the success of efficient tree-based methods for classification and regression of discrete data. We define a tree-structured permutation (TSP) that compactly encodes a permutation of discrete data where the inverse is easy to compute. We propose a decision tree algorithm to build TSPs that learns the tree structure and permutations at each node via novel criteria. We empirically demonstrate the feasibility of our method on multiple datasets. 

# Dataset paths
To run this notebook, you need to specify the correct paths to the datasets used in exp COP-H, COP-M, COP-W and mushroom, and SNPS under: \
coup_4_2_high_corr = './data/coup_data_4_2.npy' #for COP-H \
coup_4_2_moderate_corr = './data/coup_data_4_2_moderate_corr.npy' #for COP-M \
coup_4_2_weak_corr = './data/coup_data_4_2_weak_corr.npy' #for COP-W \
mushroom_data_path = './data/agaricus-lepiota.data' #for mushroom exp \
snps_805 = './data/805_SNP_1000G_real.hapt.zip' #for SNP 

To download Mushroom: Go to https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/ and download agaricus-lepiota.data and place it in the data folder 

To download SNPs Go to: https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/tree/master/1000G_real_genomes and download  805_SNP_1000G_real.hapt.zip and place it in the data folder 

# Requirements
The requirements are specified in requirements.txt \
It's advisable that you create a virtual environment and install the requirements there 

# Running the exps one at a time 
After specifying the correct paths, and activating the virtual environment you can run all the experiments like so: \
python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=3 --kfolds=5 --split_type=single_random 

exp : name of exp to run, values can be COPH, COPM, COPW, 8Gaussian, Mushroom, MNIST, SNP \
num_TSPs : The number of TSPs \
max_depth : The max depth of each TSP \
kfolds : The number of folds \
split_type : The type of split either single_random or greedy_local_perm 

There are additional parameters that you can set, like SAMPLE_AT_NODE which takes some samples at each node instead of calculating the score for all the data at each, this is particularly useful when using GLP criteria with large datasets like MNIST or SNP 

Each experiment run will generate an output file under fold_results folder 

To find the best result for a specific exp that you ran, you can run: \
python find_best_result.py --exp=COPH --kfolds=5 --split_type=single_random  

This will find the best result for COPH exp that was previously ran with 5 folds and a split type single_random 
 
# Running all exps in the paper 
chmod +x run_glp <br />
chmod +x run_rnd <br />
chmod +x find_best_glp <br /> 
chmod +x find_best_rnd <br />
./run_rnd <br />
./find_best_rnd <br /> 
./run_glp <br />
./find_best_glp  

# Code Structure  
- test_dtf.py : main function for running the different experiements and specifying the parameteres 
- find_best_result : To process the output of different depths of trees and find the best result  
- run_glp.sh : To run all exp that use GLP mentioned in paper (running this will take time, to run it: chmod +x run_glp.sh then ./run_glp.sh) 
- find_best_glp.sh : script to find best results (only run it after running run_glp.sh) 
- run_rnd.sh : To run all exp that use RND mentioned in paper (running this will take some time, but not like run_glp.sh) 
- find_best_rnd.sh : script to find best results (only run it after running run_rnd.sh) 
- DTF/TSP_structure.py : has the class definition for TSP trees and nodes 
- DTF/TSP_construction.py : contains the necessairy function to construct a TSP 
- DTF/helper_functions.py : contains helper functions that are needed when constructing TSPs 
- utilities/likelihood_functions.py : contains functions for calculating the likelihood of the data 
- utilities/plotting_functions.py : contains functions for plotting some results 
- utilities/preprocessing_functions.py : contains functions for preprocessing some datasets 
- utilities/postprocessing_functions.py : contains functions for postprocessing the results  
- utilities/printing_functions.py : contains functions for printing the results in readable form 
- utilities/sampling_functions.py : contains functions for sampling from the learned distrbuition  
- utilities/train_test_splits.py : contains functions for splitting the data to train and test 
- data/ : contains the datasets  


