import numpy as np

def write_to_report(output_filename,line):
    with open(output_filename, 'a') as f:
        f.write(line)
        f.close()  
        
def print_results(exp_results,exps_to_run,USE_pseudo_counts,SAMPLE_AT_BEGIN,SAMPLE_AT_NODE):
  for e,exp_result in enumerate(exp_results):
    exp_num = exps_to_run[e]
    for w_s in exp_result.keys():
      w_s_run = exp_result[w_s]
      
      num_ts = w_s_run.keys()
      
      #num_ts = exp_result.keys()
      for num_t in num_ts:
        max_depths = w_s_run[num_t].keys()
        for max_d in max_depths:
          avg_train_nll_bt,avg_train_nll,\
          avg_test_nll_bt,avg_test_nll,\
          avg_train_time,avg_test_time,\
          std_train_nll,std_test_nll,\
          std_train_time,std_test_time = w_s_run[num_t][max_d]
          print("************************************************************************************************************")
          print("For exp number = "+str(exp_num)+" For num_trees = "+str(num_t)+" and for max_depth = "+str(max_d))
          print("************************************************************************************************************")

          print("Results for splitting strategy = "+str(w_s))
          print("use_pseudo_counts = "+str(USE_pseudo_counts))
          print("sample_at_begin = "+str(SAMPLE_AT_BEGIN))
          print("sample_at_node = "+str(SAMPLE_AT_NODE))
          print("----------------")
          print("Training results")
          print("----------------")
          print("BT:AVG TRAIN NLL of original data is "+str(round(avg_train_nll_bt,4)))
          print("AT:AVG TRAIN NLL of data after training is "+str(round(avg_train_nll,4)))
          print("STD over train NLL across folds is "+str(round(std_train_nll,4)))
          print("AVG TRAIN TIME is "+str(round(avg_train_time,4)))
          print("STD of train time across folds is "+str(round(std_train_time,4)))
          print("----------------")
          print("Testing results")
          print("----------------")
          print("BT:AVG TEST NLL of original data is "+str(round(avg_test_nll_bt,4)))
          print("AT:AVG TEST NLL of data after training is "+str(round(avg_test_nll,4)))
          print("STD over test NLL across folds is "+str(round(std_test_nll,4)))
          print("AVG TEST TIME is "+str(round(avg_test_time,4)))
          print("STD of test time across folds is "+str(round(std_test_time,4)))
          
