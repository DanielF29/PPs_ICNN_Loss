# test running PPs with ICNN score 
##############################################################################################################################################
GPUID=$1
#datasets=("./datasets/Mix/" "./datasets/Sec/" "./datasets/Sur/")
#datasets=("./datasets/SurMix_crossV_fold1/")
#data_path="./datasets/Mix_crossV_fold1/"
data_path="./datasets/Mix/"
NUM_CLASSES=6
#Models=("resnet50" "densenet201" "vgg16")
#Models=("resnet50")
nn_model="resnet50"
#PPs_ARRAY=(1 3 10 50 100)
PPs=3
#loss= "ce, pps_loss, icnn, Gaffinity, pps_plus_icnn"
loss="pps_plus_icnn"
#experiment_counter_array=(1 2 3 4 5)
experiment_counter=1
#DA_array=("OsDA" "NoDA")
DA="OsDA"
#RUNs_ARRAY=("run1" "run2" "run3" "run4" "run5")
#RUNs_ARRAY=("run1" "run2" "run3")
RUN="pps_plus_icnn_run0"
echo "training:"
#for RUN in "${RUNs_ARRAY[@]}"; do
#	#for DA in "${DA_array[@]}"; do
#	#	for data_path in "${datasets[@]}"; do
#	#		for PPs in "${PPs_ARRAY[@]}"; do
#	#			for nn_model in "${Models[@]}"; do
echo "experiment_counter: "$experiment_counter
((experiment_counter++))
python main_Losses_ProtoPNet.py -dataset=$data_path -gpuid=$GPUID \
-base_architecture=$nn_model -da=$DA -pps_per_class=$PPs -num_classes=$NUM_CLASSES \
-run=$RUN -losses=$loss
#if [ $? -ne 0 ]; then
#	echo "Error occurred during execution of the script."
#	exit 1
#fi					
#				done
#			done
#		done
#	done
#done
