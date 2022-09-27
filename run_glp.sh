#!/bin/bash
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=2 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=3 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=4 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=5 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=6 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=7 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPH --num_TSPs=10 --max_depth=8 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=2 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=3 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=4 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=5 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=6 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=7 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPM --num_TSPs=10 --max_depth=8 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=2 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=3 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=4 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=5 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=6 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=7 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=COPW --num_TSPs=10 --max_depth=8 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=2 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=3 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=4 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=5 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=6 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=7 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=Mushroom --num_TSPs=10 --max_depth=8 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=2 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=3 --kfolds=5 --split_type=greedy_local_perm 
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=4 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=5 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=6 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=7 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=8Gaussian --num_TSPs=10 --max_depth=8 --kfolds=5 --split_type=greedy_local_perm
taskset -c 1 python test_dtf.py --exp=MNIST --num_TSPs=30 --max_depth=3 --kfolds=3 --split_type=greedy_local_perm --SAMPLE_AT_NODE=True 
taskset -c 1 python test_dtf.py --exp=MNIST --num_TSPs=30 --max_depth=7 --kfolds=3 --split_type=greedy_local_perm --SAMPLE_AT_NODE=True
taskset -c 1 python test_dtf.py --exp=MNIST --num_TSPs=30 --max_depth=10 --kfolds=3 --split_type=greedy_local_perm --SAMPLE_AT_NODE=True
taskset -c 1 python test_dtf.py --exp=SNP --num_TSPs=20 --max_depth=3 --kfolds=3 --split_type=greedy_local_perm --SAMPLE_AT_NODE=True 
