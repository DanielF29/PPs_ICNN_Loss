# Code to print the end of a little block of ProtoPNets trainings
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-pps_per_class', nargs=1, type=int) 
    parser.add_argument('-experiment_run', nargs=1, type=str)
    # python printsBeginning.py -pps_per_class=10 -experiment_run=001
    args = parser.parse_args()

    pps_per_class= args.pps_per_class[0]
    experiment_run = args.experiment_run[0]

    print('-------------------------- Beginning of training with: --------------------------')
    print('pps_per_class = ', pps_per_class)
    print('experiment_run = ', experiment_run)
    print('---------------------------------------------------------------------------')


