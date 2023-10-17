##### MODEL AND DATA LOADING for evaluation on testset
import torch
import torch.utils.data
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from PIL import Image
#import re
import os
#import copy
import pandas as pd
import torch.nn.functional as F
#from scipy import stats
#from preprocess import mean, std
from sklearn import metrics
### Visualize CFM (confusion matrix) using a heatmap
import seaborn as sns
#from scipy.stats import multivariate_normal
from helpers import makedir#, find_high_activation_crop
import model
#import train_and_test as tnt
#from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
import time
from datetime import datetime

from mod_values import bright_lvl, contr_lvl, sat_lvl, hue_lvl, A, w, texture_h

#-----------------------------------------------------------
#-----------------------------------------------------------
def forward_particular_patch(ppnet, x, nearest_patch_indices):
    distances = ppnet.prototype_distances(x) #shape (5,2000,7,7)
    patch_distances = distances.view(-1, ppnet.num_prototypes, distances.shape[2]*distances.shape[3])[:, range(distances.shape[1]), nearest_patch_indices] #shape (5,2000)
    prototype_similarities = ppnet.distance_2_similarity(patch_distances)
    logits = ppnet.last_layer(prototype_similarities)
    return logits, patch_distances, prototype_similarities

def forward_get_patch_index(ppnet, x):
    distances, conv_features = ppnet.prototype_distances(x) 
    # global min pooling
    min_distances, nearest_patch_indices = F.max_pool2d(-distances,
                                    kernel_size=(distances.size()[2],
                                                distances.size()[3]), return_indices=True)
    min_distances = -min_distances.view(-1, ppnet.num_prototypes) #shape (bs, 2000)
    nearest_patch_indices = nearest_patch_indices.view(ppnet.num_prototypes)
    prototype_similarities = ppnet.distance_2_similarity(min_distances) #shape (1,2000)
    logits = ppnet.last_layer(prototype_similarities) #shape(1,200)
    return logits, min_distances, prototype_similarities, nearest_patch_indices, conv_features

def date_string():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    # Replace "/" with "_"
    #dt_string = dt_string.replace('/', '_')
    
    return dt_string
#-----------------------------------------------------------
#-----------------------------------------------------------
# code begings
"""
# Input parameters from terminal:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str) #'./datasets/JE_128px_mixed_6classes/test_cropped/'
parser.add_argument('-csv_file_name', nargs=1, type=str) # 'KSYE_Section_2k_ppnVGG19bn_logits_df.csv'
#parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
args = parser.parse_args()
#"""
#      -----------------------------------------------------------
#      -----------------------------------------------------------
pd.set_option('display.max_columns', None)

mod_value_dict = dict()
mod_value_dict["contrast"]=contr_lvl
mod_value_dict["saturation"]=sat_lvl
mod_value_dict["hue"]=hue_lvl
mod_value_dict["shape"]=str(A)+"_"+str(w)
mod_value_dict["texture"]=texture_h
mod_value_dict["brightness"]=bright_lvl

print("mod values: ", mod_value_dict)

def def_get_performance_csv_PPs(gpuid, load_model_dir, load_model_name, test_dir, csv_file_name):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpuid # args.gpuid[0]

    # load the model
    #load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
    #load_model_name = args.model[0] #'10_18push0.7822.pth'

    # specify the test dataset folder to be analyzed
    #test_dir = args.imgdir[0] #'./datasets/JE_128px_mixed_6classes/test_cropped/' #'./local_analysis/Painted_Bunting_Class15_0081/'
    #test_dir_name = test_dir.split('/')[3]
    test_dir_name = test_dir.split('/')[-1]

    #model_base_architecture = load_model_dir.split('/')[2:]
    #print("\n def_get_performance_csv_PPs:")
    #print("load_model_dir: ", load_model_dir)
    model_base_architecture = load_model_dir.split('/')[5:]
    #experiment_run = '/'.join(load_model_dir.split('/')[3:])


    #save_analysis_path = os.path.join(test_dir, model_base_architecture, experiment_run, load_model_name)
    save_analysis_path = os.path.join('pps_results_and_descriptors_'+ test_dir_name, model_base_architecture[0]) # os.path.join('pps_explanations', model_base_architecture, experiment_run, load_model_name)
    for next_folder_name in range(len(model_base_architecture) -1 ):
        #save_analysis_path = os.path.join(save_analysis_path, model_base_architecture[next_folder_name + 1])
        save_analysis_path = save_analysis_path+"_"+model_base_architecture[next_folder_name + 1]
    makedir(save_analysis_path)
    # Name of the "csv" file to save
    #csv_file_name = args.csv_file_name[0] #'./saved_models/vgg19/003/'

    #log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    load_model_path = os.path.join(load_model_dir, load_model_name)
    #epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    #      -----------------------------------------------------------
    #      -----------------------------------------------------------

    #Previous 
    #test_dir = './' # Path to dataset
    #test_dataset = 'test' #previously used
    # model name (on the same folder as this script)
    # 99nopush0.9921.pth

    #load_model_path = "./99nopush0.9921.pth"
    ppnet = torch.load(load_model_path)
    #print(ppnet)

    #if torch.cuda.is_available():
    #   ppnet = ppnet.cuda()
    ### ppnet = ppnet.cpu()
    # set evaluate mode for the ppnet model:
    ppnet.eval()

    # Get network properties
    img_size = ppnet.img_size  # Image size
    #prototype_shape = ppnet.prototype_shape # Prototype shape
    #max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    total_num_pps = ppnet.num_prototypes
    num_classes = ppnet.num_classes

    # Initialize preprocessing function used for prototypes
    preprocess = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])

    # Make dataframe for results (found difference)
    #num_classes = 6
    logits_df = pd.DataFrame(columns=["F"+str(i) for i in range(0, num_classes + 4)])
    logits_df = logits_df.rename(columns={'F0': 'subdir', 'F1': 'filename', 'F2': 'predicted_cls','F3': 'correct_cls'})

    with torch.no_grad():
        #out_data = model(data)
        prototypes = range(total_num_pps) #range(60) # Prototype indices (assumes 60 prototypes for El Beze KS dataset (6 classes, 10 PPs per class))
        #max_count = 2
        count=0
        # Loop through image files in all image folders
        for path, subdirs, files in os.walk(test_dir): #for path, subdirs, files in os.walk(os.path.join(test_dir, test_dataset)):
            for subdir in subdirs: #This second "for" loop, go through the class folders in the "test" folder
                count+=1
                subdir_subtype = subdir.split("_")[-1]
                print("class: ", subdir, count, "/", num_classes, flush=True)
                """ 
                if subdir == "MIX-Subtype_Ia":
                    labels_test = torch.tensor([0])
                if subdir == "MIX-Subtype_IIa":
                    labels_test = torch.tensor([1])    
                if subdir == "MIX-Subtype_IIIa":
                    labels_test = torch.tensor([2])
                if subdir == "MIX-Subtype_IVc":
                    labels_test = torch.tensor([3]) 
                if subdir == "MIX-Subtype_IVd":
                    labels_test = torch.tensor([4])
                if subdir == "MIX-Subtype_Va":
                    labels_test = torch.tensor([5])
                """
                if subdir_subtype == "Ia":
                    labels_test = torch.tensor([4])
                if subdir_subtype == "IIa":
                    labels_test = torch.tensor([1])    
                if subdir_subtype == "IIIa":
                    labels_test = torch.tensor([0])
                if subdir_subtype == "IVc":
                    labels_test = torch.tensor([2]) 
                if subdir_subtype == "IVd":
                    labels_test = torch.tensor([3])
                if subdir_subtype == "Va":
                    labels_test = torch.tensor([5])      

                for class_path, class_subdirs, class_files in os.walk(os.path.join(test_dir,subdir)): #for class_path, class_subdirs, class_files in os.walk(os.path.join(os.path.join(test_dir, test_dataset),subdir)):
                    # Loop through files in folder 
                    for filename in class_files:
                        img_path = os.path.join(class_path, filename) # Get path of file
                        mod_tensors = []
                        # Open image and convert to RGB
                        try:
                            img_pil = Image.open(img_path).convert('RGB')
                        except:
                            img_path = img_path + '.png' #'.jpg'
                            img_pil = Image.open(img_path).convert('RGB')
                        image_orig = preprocess(img_pil).unsqueeze(0)   # Apply preprocessing function
                        if torch.cuda.is_available():
                            image_orig = image_orig.cuda() # Utilize GPU
                        # Get network output
                        with torch.no_grad():
                            logits, min_distances, orig_similarities, nearest_patch_indices, conv_features = forward_get_patch_index(ppnet, image_orig)

                        ########################################
                        flat_features = conv_features.view(conv_features.size(0), -1)
                        flat_features_np = flat_features.cpu().detach().numpy()
                        ########################################

                        #print(logits)
                        #print(logits.size())
                        #print(logits.size(dim=1))
                        #
                        tables = []
                        for i in range(logits.size(0)):
                            tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))  

                        #arr = np.empty(10)
                        #arr[0] = class_path
                        #arr[1] =  filename
                        #arr[2] =  tables[0][0]  # predicted_cls
                        #arr[3] =  tables[0][1]  # correct_cls
                        #for logit_index in range(logits.size(dim=1)):
                        #    arr[4+logit_index] = logits[0,logit_index]

                        logits_size = logits.size(dim=1)
                        df_lenght = 4 + logits_size + len(flat_features_np)
                        df = pd.DataFrame(columns=["F"+str(i) for i in range(0, df_lenght)])
                        df = df.rename(columns={'F0': 'subdir', 'F1': 'filename', 'F2': 'predicted_cls','F3': 'correct_cls'})

                        #df.at['subdir'] = subdir
                        df['subdir'] = subdir
                        #print(subdir)
                        #print(df)
                        df['filename'] =  filename
                        #print(filename)
                        #print(df)
                        df['predicted_cls'] =  tables[0][0]
                        #print(tables[0][0])
                        #print(df)
                        df['correct_cls'] =  tables[0][1]
                        #print(tables[0][1])
                        #print(df)
                        for logit_index in range(logits_size):
                            df["F"+str(4+logit_index)] = logits[0,logit_index].item()

                        for flat_features_index in range(len(flat_features_np)):
                            df["F"+str(4+logits_size+flat_features_index)] = flat_features_np[flat_features_index] #.item()

                        #print(df)
                        #print(arr)

                        # Put row in total results
                        #logits_df = logits_df.append(pd.DataFrame(arr.reshape(1,-1), columns=list(logits_df)), ignore_index=True)
                        #logits_df = logits_df.append(df, ignore_index=True)
                        logits_df= pd.concat([logits_df, df])
                        # # # df.to_csv('conv_features.csv', mode='a', header=False, index=False)  # mode='a' for appending
                        #print("")
                        #print(logits_df)
                        #break
                    #break
                #break
            #break

    path_to_csv_file = os.path.join(save_analysis_path, csv_file_name + ".csv") # generates the path and name for the "csv" file
    #saves the "csv" file with the description, file name /img name, the predicted and groundtrhut class, as well as the output logits
    logits_df.to_csv(path_to_csv_file, index=False) # logits_df.to_csv('KSYE_Section_2k_ppnVGG19bn_logits_df.csv', index=False)

    ###
    ### Evaluation of some simple performance metrics
    ### On results from a model read as an array.
    ###

    print("Reading and evaluating ...:" + csv_file_name)
    data = np.genfromtxt(path_to_csv_file, delimiter=',', skip_header=1)
    #data = np.delete(data, 0, 0)
    #data = np.delete(data, [0,1,4,5,6,7,8,9], 1)
    data = data[:, [2, 3]]

    #"""
    ## Evaluation of Accuracy on test set from scratch
    def accuracy(y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy
    #"""
    y = data[:,1]
    y_pred = data[:,0]
    print("Accuracy on  train: ", accuracy(y, y_pred )  )


    #Confusion Matrix:
    #Evaluate model using scikit learn "metrics" to
    # obtain the confusion matrix from pairs of predictions and ground truth labels.
    # This is basically looking at how well your model did on predictions
    cnf_matrix = metrics.confusion_matrix(y, y_pred)
    print(cnf_matrix)
    cnf_matrix_df = pd.DataFrame(cnf_matrix)
    #cnf_matrix_df.to_csv('Cnf_Matrix_{csv_file_name}.csv', index=False, header=False)
    path_to_Cnf_Matrix_csv_file = os.path.join(save_analysis_path, 'Cnf_Matrix_' + csv_file_name + '.csv') # generates the path and name for the "csv" file
    cnf_matrix_df.to_csv(path_to_Cnf_Matrix_csv_file, index=False, header=False)

    ###
    ### Visualize CFM (confusion matrix) using a heatmap
    ###
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues" ,fmt='g') #another color map for confusion matrix: YlGnBu
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    Cnf_Matrix_img_path = os.path.join(save_analysis_path, 'Cnf_Matrix_' + csv_file_name + '.png')
    plt.savefig(Cnf_Matrix_img_path)
    plt.close()

    ###
    ### Evaluation metrics Accuracy, Precision, Recall
    ###
    model_accuracy =  metrics.accuracy_score(y, y_pred)
    model_precision =  metrics.precision_score(y, y_pred, average='weighted')
    model_recall =  metrics.recall_score(y, y_pred, average='weighted')
    model_F1   =  metrics.f1_score(y, y_pred, average='weighted')

    print("Accuracy:  ",  model_accuracy)
    print("Precision: ", model_precision)
    print("Recall:    ",    model_recall)
    print("F1:        ",        model_F1) 

    Eval_Metrics_path = os.path.join(save_analysis_path, 'Eval_Metrics_' + csv_file_name + load_model_name.split(".")[0] + '.txt')
    with open(Eval_Metrics_path, 'a') as f:
        f.write("Accuracy:  " + str(model_accuracy)  + '\n')
        f.write("Precision: " + str(model_precision) + '\n')
        f.write("Recall:    " + str(model_recall)    + '\n')
        f.write("F1:        " + str(model_F1)        + '\n')
    f.close

    return

""" 
if __name__ == "__main__":
    start = time.time()
    dt_string = date_string()
    #--------------------------#
    gpuid="0"
    load_model_dir  = "/workspace/ICNN_PPs_results/Mix_results/saved_models/run1/Mix_OsDA_resnet50_1/"
    load_model_name = "30_13push_0.8975.pth"
    test_dir = "/workspace/ICNN_PPs_results/Mix_results/datasets/test/"
    csv_file_name = "test.csv"
    def_get_performance_csv_PPs( 
        gpuid, 
        load_model_dir,
        load_model_name, 
        test_dir, 
        csv_file_name
        )
#"""

