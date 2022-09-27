#!/bin/bash
taskset -c 1 python find_best_result.py --exp=COPH --kfolds=5 --split_type=single_random 
taskset -c 1 python find_best_result.py --exp=COPM --kfolds=5 --split_type=single_random 
taskset -c 1 python find_best_result.py --exp=COPW --kfolds=5 --split_type=single_random
taskset -c 1 python find_best_result.py --exp=Mushroom --kfolds=5 --split_type=single_random
taskset -c 1 python find_best_result.py --exp=8Gaussian --kfolds=5 --split_type=single_random
taskset -c 1 python find_best_result.py --exp=MNIST --kfolds=3 --split_type=single_random
taskset -c 1 python find_best_result.py --exp=SNP --kfolds=3 --split_type=single_random 
