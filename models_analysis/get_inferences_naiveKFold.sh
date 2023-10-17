# sh file to run inferences on ProtoPNet models and save the "csv" file
#
gpu=$1
test_Dataset='../datasets/Mix_naive_KFoldTrainVal_test/test/'
#
#1
python get_performance_csv_PPs_naive_KFold.py \
-gpuid=$gpu \
-modeldir="../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold0/" \
-model="20push_0.9974.pth" \
-imgdir=$test_Dataset
#2
python get_performance_csv_PPs_naive_KFold.py \
-gpuid=$gpu \
-modeldir="../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold1/" \
-model="10push_0.9958.pth" \
-imgdir=$test_Dataset
#3
python get_performance_csv_PPs_naive_KFold.py \
-gpuid=$gpu \
-modeldir="../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold2/" \
-model="10push_0.9958.pth" \
-imgdir=$test_Dataset
#4
python get_performance_csv_PPs_naive_KFold.py \
-gpuid=$gpu \
-modeldir="../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold3/" \
-model="20push_0.9979.pth" \
-imgdir=$test_Dataset
#5
python get_performance_csv_PPs_naive_KFold.py \
-gpuid=$gpu \
-modeldir="../saved_models/CrossV_run1/Mix_naive_KFoldTrainVal_test_OsDA_resnet50_3pps_Fold4/" \
-model="10push_0.9964.pth" \
-imgdir=$test_Dataset
#
