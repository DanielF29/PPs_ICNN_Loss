# Code to get_Accuracy_evaluation_result for each model tested 
# and save those results in a csv
import os
import time
from datetime import datetime
import pandas as pd
import re

def get_directories(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_bottom_directories(path):
    dirs = get_directories(path)
    if not dirs:
        return [path]
    bottom_dirs = []
    for d in dirs:
        bottom_dirs.extend(get_bottom_directories(d))
    return bottom_dirs

def get_all_bottom_subfolders():
    bottom_dirs = get_bottom_directories(os.getcwd())
    for d in bottom_dirs:
        print(d)

def get_depth_directories(path, current_depth, target_depth):
    if current_depth == target_depth:
        return [path]
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    subdirs = []
    for d in dirs:
        subdirs.extend(get_depth_directories(d, current_depth + 1, target_depth))
    return subdirs

def get_list_of_push_models_accu(model_path):
    files = os.listdir(model_path) 
    list_of_push_models_accu = []
    for file in files:
        if (not os.path.isdir(os.file.join(model_path, file)) ):
            if (file.lower().split(".")[-1] == "pth") and (not "nopush_" in file):
                accu_string = file.lower().split("push_")[-1] 
                accu_string = accu_string.split(".")[1]
                list_of_push_models_accu.append(int(accu_string))
    return list_of_push_models_accu

def extract_number(filename):
    try:
        reconstruct_num = int(''.join([c for c in filename if c.isdigit()])) 
                                                            # c.isdigit() only return True for single integers, not points
        #print(reconstruct_num)
        return reconstruct_num
    except:
        return 0

def date_string():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    # Replace "/" with "_"
    #dt_string = dt_string.replace('/', '_')
    
    return dt_string

######################################################################################
######################################################################################
start = time.time()
dt_string = date_string()
save_accu_csv_path = "./"

# We will get all the folders paths at a depth of "target_depth = 2"
target_depth = 2
explored_folder_path = "../saved_models/" # + os.getcwd()
subdirs = get_depth_directories(explored_folder_path, 0, target_depth)
#num_subfolders = len(subdirs)
#print(subdirs)

# Now we will extract the path to the files containing the accuracy results
# on the previous folder paths, those paths are expected to caontain "pps_results_test" 
# and the file names to begin with "Eval_Metrics_test_"
num_subfolders = 0
num_rows = 0
#accu_files_list = []
main_df = pd.DataFrame()
for d in subdirs:
    #print(d)
    if (("run6" in d) or ("run7" in d) or ("run8" in d) or ("run9" in d) or ("run10" in d)):
        num_subfolders += 1
        for f_path in os.listdir(d):
            #print(f_path)
            if (f_path.lower().split(".")[-1] == "pth") and (not "nopush_" in f_path):
                num_rows += 1
                # Extraction of the details to be registered 
                #   Accuracy from the files name
                model_name = f_path
                accu_string = f_path.lower().split("push_")[-1] 
                accu_string = accu_string.split(".")[1]
                read_accu = int(accu_string)/100.0
                #print(read_accu)
                #   The rest of the model, training and test details from the path
                folder_name_parts = d.split("\\")
                run_num = folder_name_parts[-2]
                run_num = run_num.split("/")[-1]
                model_details = folder_name_parts[-1].split("_")
                train_data = model_details[0]
                test_data = train_data
                train_DA = model_details[1]
                backbone = model_details[2]
                pps_num = model_details[3]
                if "pps" in pps_num:
                    pps_num = pps_num.split("pps")[0]
                complete_f_path = os.path.join(d, f_path)
                #print(complete_f_path)
                
                # Creating dictionary from all the details extracted (Accuracy, model, training and test details)
                model_details_dict = {'Accuracy': read_accu, \
                                    'train_data': train_data, \
                                    'Data_Augmentation':train_DA, \
                                    'test_data':test_data, \
                                    'run_number':run_num, \
                                    'pps_number':pps_num, \
                                    'model_name':model_name, \
                                    'backbone':backbone, \
                                    'complete_path':complete_f_path
                                    }

                # Convert the dictionary to a pandas DataFrame
                df_single_model_data = pd.DataFrame([model_details_dict])
                # Concat the "df_single_model_data" DataFrame to the "main DataFrame"
                main_df = pd.concat([main_df, df_single_model_data])

print("num_subfolders: ", num_subfolders)
print("num_rows: ", num_rows)
print("main_df: ")
print(main_df)
dt_string = date_string()
main_df.to_csv(dt_string + '_PPs_icnn_models_original_test_evals.csv', index=False)

end = time.time()
lapsed_time = end -  start
print("start: ", start)
print("end: ", end)
print("lapsed_time: ", lapsed_time)
######################################################################################
######################################################################################

