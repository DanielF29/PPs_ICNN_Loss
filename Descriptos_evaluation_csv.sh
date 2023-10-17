# sh file to evaluate descriptors of ProtoPNet models and save the "csv" file 
#################################################################################
########   No Data Augmentation
#
#          DenseNet201
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/densenet201_006_pps1_NoDA/ \
-model=10_12push_0.8192.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_001pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/densenet201_015_pps3_NoDA/ \
-model=10_9push_0.8733.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_003pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/densenet201_024_pps10_NoDA/ \
-model=20_12push_0.8229.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_010pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/densenet201_033_pps50_NoDA/ \
-model=20_19push_0.8200.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_050pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/densenet201_042_pps100_NoDA/ \
-model=30_19push_0.8696.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_100pps_noDA
#
#          ResNet50
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/resnet50_003_pps1_NoDA/ \
-model=10_19push_0.8454.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_001pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/resnet50_012_pps3_NoDA/ \
-model=30_19push_0.8588.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_003pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/resnet50_021_pps10_NoDA/ \
-model=20_18push_0.8292.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_010pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/resnet50_030_pps50_NoDA/ \
-model=10_19push_0.8208.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_050pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/resnet50_039_pps100_NoDA/ \
-model=30_19push_0.8037.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_100pps_noDA
#
#          Vgg16
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/vgg16_009_pps1_NoDA/ \
-model=10_19push_0.7696.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_001pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/vgg16_018_pps3_NoDA/ \
-model=30_19push_0.7913.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_003pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/vgg16_027_pps10_NoDA/ \
-model=30_19push_0.7608.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_010pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/vgg16_036_pps50_NoDA/ \
-model=10_19push_0.7671.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_050pps_noDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_NoDA_w256/vgg16_045_pps100_NoDA/ \
-model=30_19push_0.7562.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_100pps_noDA
#
####################################################################################
####################################################################################
########   JE_Mix_6c_OscarDA_w256
#
#          DenseNet201
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/densenet201_006_pps1_OsDA/ \
-model=20_19push_0.8296.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_001pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/densenet201_015_pps3_OsDA/ \
-model=30_19push_0.8729.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_003pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/densenet201_024_pps10_OsDA/ \
-model=10_9push_0.8767.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_010pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/densenet201_033_pps50_OsDA/ \
-model=20_19push_0.8483.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_050pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/densenet201_042_pps100_OsDA/ \
-model=30_19push_0.8458.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=DenseNet201_JE_6c_Mix_w256_100pps_OsDA
#
#          ResNet50
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/resnet50_003_pps1_OsDA/ \
-model=30_8push_0.8854.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_001pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/resnet50_012_pps3_OsDA/ \
-model=30_19push_0.8529.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_003pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/resnet50_021_pps10_OsDA/ \
-model=30_19push_0.8350.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_010pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/resnet50_030_pps50_OsDA/ \
-model=30_19push_0.8638.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_050pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/resnet50_039_pps100_OsDA/ \
-model=20_19push_0.8588.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=ResNet50_JE_6c_Mix_w256_100pps_OsDA
#
#          Vgg16
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/vgg16_009_pps1_OsDA/ \
-model=30_19push_0.8267.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_001pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/vgg16_018_pps3_OsDA/ \
-model=30_19push_0.8383.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_003pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/vgg16_027_pps10_OsDA/ \
-model=10_13push_0.8358.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_010pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/vgg16_036_pps50_OsDA/ \
-model=20_19push_0.8058.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_050pps_OsDA
#
python descriptors_evaluation_pps_act_logs.py \
-gpuid=0 \
-modeldir=./PPs/JE_Mix_6c_w256/JE_Mix_6c_OscarDA_w256/vgg16_045_pps100_OsDA/ \
-model=10_19push_0.7433.pth \
-imgdir=./datasets/Mix/ \
-csv_file_name=Vgg16_JE_6c_Mix_w256_100pps_OsDA
#