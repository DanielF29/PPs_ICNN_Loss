# Code to print the end of a little block of ProtoPNets trainings
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_architecture', nargs=1, type=str) 
    parser.add_argument('-pps_per_class', nargs=1, type=int) 
    parser.add_argument('-num_classes', nargs=1, type=int)
    parser.add_argument('-experiment_run', nargs=1, type=str)
    # python printsEnd.py -base_architecture=resnet50 -pps_per_class=10 -num_classes=6 -experiment_run=003
    args = parser.parse_args()

    base_architecture = args.base_architecture[0]
    pps_per_class= args.pps_per_class[0]
    num_classes= args.num_classes[0]
    experiment_run = args.experiment_run[0]

    print('-------------------------- End of training with: --------------------------')
    print('base_architecture = ', base_architecture)
    print('pps_per_class = ', pps_per_class)
    print('num_classes = ', num_classes)
    print('experiment_run = ', experiment_run)
    print('---------------------------------------------------------------------------')


