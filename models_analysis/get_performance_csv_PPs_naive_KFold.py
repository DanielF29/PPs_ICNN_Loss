##### MODEL AND DATA LOADING for evaluation on testset
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import re
import os
import copy
import pandas as pd
import torch.nn.functional as F
from scipy import stats
#from preprocess import mean, std
from sklearn import metrics
### Visualize CFM (confusion matrix) using a heatmap
import seaborn as sns
#from scipy.stats import multivariate_normal
from helpers import makedir, find_high_activation_crop
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

#-----------------------------------------------------------
#-----------------------------------------------------------
def forward_particular_patch(ppnet, x, nearest_patch_indices):
    distances = ppnet.prototype_distances(x) #shape (5,2000,7,7)
    patch_distances = distances.view(-1, ppnet.num_prototypes, distances.shape[2]*distances.shape[3])[:, range(distances.shape[1]), nearest_patch_indices] #shape (5,2000)
    prototype_similarities = ppnet.distance_2_similarity(patch_distances)
    logits = ppnet.last_layer(prototype_similarities)
    return logits, patch_distances, prototype_similarities

def forward_get_patch_index(ppnet, x):
    distances = ppnet.prototype_distances(x) 
    # global min pooling
    min_distances, nearest_patch_indices = F.max_pool2d(-distances,
                                    kernel_size=(distances.size()[2],
                                                distances.size()[3]), return_indices=True)
    min_distances = -min_distances.view(-1, ppnet.num_prototypes) #shape (bs, 2000)
    nearest_patch_indices = nearest_patch_indices.view(ppnet.num_prototypes)
    prototype_similarities = ppnet.distance_2_similarity(min_distances) #shape (1,2000)
    logits = ppnet.last_layer(prototype_similarities) #shape(1,200)
    return logits, min_distances, prototype_similarities, nearest_patch_indices
#-----------------------------------------------------------
#-----------------------------------------------------------
# code begings
# Input parameters from terminal:
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # -gpuid=0,1,2,3
parser.add_argument('-modeldir', nargs=1, type=str) # -modeldir='../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0/'
parser.add_argument('-model', nargs=1, type=str) #-model="20push_0.9974.pth"
parser.add_argument('-imgdir', nargs=1, type=str) #'../datasets/Mix_naive_KFoldTrainVal_test/test/'
#parser.add_argument('-csv_file_name', nargs=1, type=str) # 'KSJE1k_w256_Mix_ResNet50_logits_df.csv'
#parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
args = parser.parse_args()
#      -----------------------------------------------------------
#      -----------------------------------------------------------
print("begin")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# load the model
load_model_dir = args.modeldir[0] #'../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0/'
load_model_name = args.model[0] #"20push_0.9974.pth"

# specify the test dataset folder to be analyzed
test_dir = args.imgdir[0] #'../datasets/Mix_naive_KFoldTrainVal_test/test/'
main_dataset_folder = test_dir.split('/')[-] # '../datasets/Mix_naive_KFoldTrainVal_test/test/' --> Mix_naive_KFoldTrainVal_test
test_dir_name = main_dataset_folder.split('_')[0] # '../datasets/Mix_naive_KFoldTrainVal_test/test/' --> "Mix"

model_base_architecture = load_model_dir.split('/')[2:] 
#'../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0/' -> ["CrossV_run1", "Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0", ""]
experiment_run = (model_base_architecture[0]).split('_')[-1]
#"CrossV_run1"->"run1"
model_details = (model_base_architecture[1]).split('_') 
#"Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0"->["Mix","naive","KFoldTrainVal","test","OsDA","resnet50","3pps","Fold0"]
#                                                           0       1           2           3      4        5        6       7

extra_descrip = "trainedOn_JE6c1kw256_" # this is a extra_train_set_description
# Name of the "csv" file to save
csv_file_name = model_details[1] +"KFold_"+ 'testsOn_'+ test_dir_name +"_"+ extra_descrip \
    +"_"+ model_details[0] +"_"+ model_details[4] +"_"+ model_details[5] +"_"+ model_details[6] +"_"+ model_details[7]
# csv_file_name = "naiveKFold_testsOn_Mix_trainedOn_JE6c1kw256_Mix_OsDA_resnet50_3pps_Fold0"
save_analysis_path = 'tests_on_'+ test_dir_name
# save_analysis_path = "tests_on_Mix"
makedir(save_analysis_path)

print("csv_file_name: ", csv_file_name)
print("save_analysis_path: ", save_analysis_path)

load_model_path = os.path.join(load_model_dir, load_model_name)
#load_model_path = '../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0/20push_0.9974.pth'
ppnet = torch.load(load_model_path)
#print(ppnet)

if torch.cuda.is_available():
    ppnet = ppnet.cuda()
# set evaluate mode for the ppnet model:
ppnet.eval()

# Get network properties
img_size = ppnet.img_size  # Image size
prototype_shape = ppnet.prototype_shape # Prototype shape
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
logits_df = pd.DataFrame(columns=["F"+str(i) for i in range(0, num_classes + 4)])
logits_df = logits_df.rename(columns={'F0': 'subdir', 'F1': 'filename', 'F2': 'predicted_cls','F3': 'correct_cls'})

with torch.no_grad():
    prototypes = range(total_num_pps) #range(60) # Prototype indices 
    count=0
    # Loop through image files in all image folders 
    for path, subdirs, files in os.walk(test_dir): #for path, subdirs, files in os.walk(os.path.join(test_dir, test_dataset)):
        for subdir in subdirs: #This second "for" loop, go through the class folders in the "test" folder
            count+=1
            print("class: ", subdir,"  ", count, "/", num_classes, flush=True)
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
            if subdir == "MIX-Subtype_Ia":
                labels_test = torch.tensor([4])
            if subdir == "MIX-Subtype_IIa":
                labels_test = torch.tensor([1])    
            if subdir == "MIX-Subtype_IIIa":
                labels_test = torch.tensor([0])
            if subdir == "MIX-Subtype_IVc":
                labels_test = torch.tensor([2]) 
            if subdir == "MIX-Subtype_IVd":
                labels_test = torch.tensor([3])
            if subdir == "MIX-Subtype_Va":
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
                        logits, min_distances, orig_similarities, nearest_patch_indices = forward_get_patch_index(ppnet, image_orig)

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

                    df = pd.DataFrame(columns=["F"+str(i) for i in range(0, logits.size(dim=1) + 4)])
                    df = df.rename(columns={'F0': 'subdir', 'F1': 'filename', 'F2': 'predicted_cls','F3': 'correct_cls'})

                    df.at['subdir'] = subdir
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
                    for logit_index in range(logits.size(dim=1)):
                        df["F"+str(logit_index+4)] = logits[0,logit_index].item()

                    #print(df)
                    #print(arr)

                    # Put row in total results
                    #logits_df = logits_df.append(pd.DataFrame(arr.reshape(1,-1), columns=list(logits_df)), ignore_index=True)
                    logits_df = logits_df.append(df, ignore_index=True)
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
data = np.delete(data, 0, 0)
data = np.delete(data, [0,1,4,5,6,7,8,9], 1)

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

Eval_Metrics_path = os.path.join(save_analysis_path, 'Eval_Metrics_' + csv_file_name + load_model_name + '.txt')
with open(Eval_Metrics_path, 'a') as f:
    f.write("Accuracy:  " + str(model_accuracy)  + '\n')
    f.write("Precision: " + str(model_precision) + '\n')
    f.write("Recall:    " + str(model_recall)    + '\n')
    f.write("F1:        " + str(model_F1)        + '\n')
f.close
