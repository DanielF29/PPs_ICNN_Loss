# Code to get evaluation of PPs models 
# on the best model at all the bottom sub-folders within an initial input folder path
import os
#import get_performance_csv_PPs
#import test_arg_parse
import def_get_performance_csv_PPs
from log import create_logger
import time
from datetime import datetime

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

def remove_previous_occurrences(lst, target):
    index = None
    for i, item in enumerate(lst):
        #print(item)
        if item == target:
            index = i
            break
    if index is not None:
        return lst[index:]
    else:
        return lst

def get_last_dir_here(path):
    #last_folder_full_path = os.path.join(path, os.listdir(path)[-1])
    return os.listdir(path)[-1]

def get_pending_evaluations_list(full_list, subtraction_list):
    difference_dictionary = {}
    pending_evaluations_list = {}
    for key in full_list:
        if key in subtraction_list:
            difference_dictionary[key] = [value for value in full_list[key] if value not in subtraction_list[key]]
            key_path = "D:/Results_trained_models/ProtoPNets/ISBI_2023/PPs/"+key[:19]+"/"+key[20:24]+"/"+key[25:]
            pending_evaluations_list[key_path] = []
            for test in difference_dictionary[key]:
                test_path = "D:/Results_trained_models/ProtoPNets/ISBI_2023/datasets/" + test[:3] + "/" + test[4:]
                pending_evaluations_list[key_path].append(test_path)
    return pending_evaluations_list

def get_evaluations_list(path):
    # Gets the list of already evaluated models, to avoid repeating evaluations
    target_depth = 2
    subdirs_level2 = get_depth_directories(path, 0, target_depth)
    #num_subfolders = 0
    #num_rows = 0
    done_evaluations_list = {}
    for d in subdirs_level2:
        #test_folder_name = d.split("\\")[5]
        if ("pps_results_test" in d) and (not "run6" in d): # if ("pps_results_test" == test_folder_name): # usung "==" limits the search within only the original testset evaluations
            #num_subfolders += 1
            model_name = d.replace('\\', '/') # Since the code is ran on a git-bash terminal adjustments to the path are made
            detailed_model_name = model_name.split('/')[-1]
            if not detailed_model_name in done_evaluations_list:
                done_evaluations_list[detailed_model_name] = []
            for f_path in os.listdir(d):
                if ("Eval_Metrics_test_" in f_path):
                    #num_rows += 1
                    splitted_f_path = f_path.split("Eval_Metrics_")[-1]
                    test_name = splitted_f_path.split("_run")[0]
                    ordered_test_name = test_name[-3:] + "_" + test_name[:-4]
                    done_evaluations_list[detailed_model_name].append(ordered_test_name)
    return done_evaluations_list


######################################################################################
######################################################################################
start = time.time()
dt_string = date_string()
# Create log file, with name based on the datetime
save_analysis_path = "./"
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, dt_string +'eval_progress.log'))
log('evaluation of PPs models:')
# setting some variables for the evaluations 
base_dir = "D:/Results_trained_models/ProtoPNets/ISBI_2023/"
first_mod_testdata = "test_brightness_0.8"
imgdir_list = [
    base_dir + "datasets/Mix/test",
    base_dir + "datasets/Mix/test_brightness_0.8", # + first_mod_testdata,
    base_dir + "datasets/Mix/test_contrast_0.45",
    base_dir + "datasets/Mix/test_grey",
    base_dir + "datasets/Mix/test_saturation_0.7",
    base_dir + "datasets/Mix/test_shape_4_0.05",
    base_dir + "datasets/Mix/test_texture_4"
    ]








target_depth = 3
subdirs = get_depth_directories(os.getcwd(), 0, target_depth)
num_subfolders = len(subdirs)
log("total_num_subfolders: {0}".format(num_subfolders))
print(subdirs[0])
# get folder were the previous training was left,
# careful with the path entered
#last_model_eval_folder = get_last_dir_here(base_dir + "PPs/pps_results_" + first_mod_testdata) 
#print(last_model_eval_folder)
#target = os.path.join(base_dir + "PPs",
#                    last_model_eval_folder.split("_run")[0],
#                    last_model_eval_folder.split("_")[5], 
#                    last_model_eval_folder.split("_run")[1][2:] 
#                    )
#target = target.replace('/', '\\')
#print(subdirs)
#log("target: {0}".format(target))
#subdirs = remove_previous_occurrences(subdirs, target)

complete_evaluations_list = {}
for model_path in subdirs:
    model_name = model_path.replace('\\', '/') # Since the code is ran on a git-bash terminal adjustments to the path are made
    details_from_model_name = model_name.split('/')[5:]
    detailed_model_name = '_'.join(details_from_model_name)
    complete_evaluations_list[detailed_model_name] = []
    for imgdir in imgdir_list:
        imgdir = imgdir.split("datasets/")[-1]
        imgdir = imgdir.replace('/','_')
        complete_evaluations_list[detailed_model_name].append(imgdir)
# Flatten the lists in the dictionary into a single list of values
all_values = [value for values in complete_evaluations_list.values() for value in values]
num_of_values = len(all_values)
print("num_of_values_on_complete_evaluations_list: ",num_of_values)

eval_so_far_list = get_evaluations_list(os.getcwd())
# Flatten the lists in the dictionary into a single list of values
all_values = [value for values in eval_so_far_list.values() for value in values]
num_of_values = len(all_values)
print("num_of_values_on_eval_so_far_list: ", num_of_values)

pending_evaluations_list = get_pending_evaluations_list(complete_evaluations_list, eval_so_far_list)
# Flatten the lists in the dictionary into a single list of values
all_values = [value for values in pending_evaluations_list.values() for value in values]
num_of_values = len(all_values)
print("num_of_values_on_pending_evaluations_list: ", num_of_values)

#key, value = next(iter(complete_evaluations_list.items()))
#print(f"{key}: {value}")
#print(complete_evaluations_list.items())
#print("----------------------------------------------------   ")
#print("----------------------------------------------------   ")
#print("----------------------------------------------------   ")
#key, value = next(iter(eval_so_far_list.items()))
#print(f"{key}: {value}")
#print(eval_list_so_far.items())
#print("----------------------------------------------------   ")
#key, value = next(iter(pending_evaluations_list.items()))
#print(f"{key}: {value}")

#subdirs = remove_previous_evaluated_models(subdirs, target)
#num_subfolders = len(subdirs)
#log('num subfolders to evaluate: {0}'.format(num_subfolders))
log(' ')
gpuid="0"
key_count = 0
tests_count = 0
num_keys = len(pending_evaluations_list) # the number of key should coincide with the number of models pending to evaluate
                                         # on at least one dataset
#"""
for key_model_path, value_test_path in pending_evaluations_list.items():
    # key_model_path is the nth key value of the dictionary and is the path to the model to be evaluated
    # value_test_path is the list of pending datasets on which to evaluate the best model in key_model_path
    key_count += 1
    log('folder_path: \t{0}'.format(key_model_path))
    progress_percentage =  "{:.2f}".format(100 * key_count / num_keys)
    log('progress: {0}/{1},\t percentage: {2}'.format(key_count, num_keys, progress_percentage))

    # Obtaining the ".pth" model with the best accuracy on its name
    files = [f for f in os.listdir(key_model_path) if ("push_" in f) and (not "nopush_" in f)]
    #print(files)
    highest_number = 0
    highest_filename = None
    for filename in files:
        number = extract_number(filename.split("push_")[-1]) # Extrae el numero antes de ".pth", ejemplo: "10_12push_0.8192.pth" -- 8192
        #print(number)
        if number > highest_number:
            highest_number = number
            highest_filename = filename

    # Evaluation of the identified best ".pth" model
    if highest_filename:
        #print("File with highest accuracy:", highest_filename)
        modeldir= key_model_path#.replace('\\', '/')
        model_name= highest_filename
        #print(highest_filename)
        #print(value_test_path)
        for imgdir in value_test_path:
            tests_count += 1
            log('Evaluation data: {0}'.format(imgdir.split("/")[-1]))
            csv_file_name= imgdir.split("/")[-1] + "_" + imgdir.split("/")[-2] + "_" + modeldir.split("/")[6] + "_" + modeldir.split("/")[7] #+ ".csv"
            def_get_performance_csv_PPs.def_get_performance_csv_PPs(gpuid, modeldir, model_name, imgdir, csv_file_name)
        
    else:
        print("No files found.")
#"""
print("----------------------------------------------------   ")
print("tests_count: ", tests_count)
print("----------------------------------------------------   ")

dt_string = date_string()
end = time.time()
log('Ending date and time: {0}'.format(dt_string))
log('\ttotal_evaluation_time: \t{0}'.format(end -  start))
logclose()

