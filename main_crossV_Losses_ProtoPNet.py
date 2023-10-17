#from ast import arg
import os
import shutil
import csv
import argparse
import re

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
#from preprocess import mean, std, preprocess_input_function
from preprocess import mean, std, preprocess_input_function
import utils.globals as global_vars


def delete_previous_models(model_path_to_delete):
    # Check if the file exists
    if os.path.exists(model_path_to_delete):
        # Delete the file
        os.remove(model_path_to_delete)
        print(f"{model_path_to_delete} deleted successfully.")
    else:
        print(f"{model_path_to_delete} does not exist.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='./datasets/Mix_crossV/')
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # -gpuid=0,1,2,3
    parser.add_argument('-base_architecture', nargs=1, type=str, default='resnet50') 
    parser.add_argument('-da', type=str, default='OsDA') 
    parser.add_argument('-pps_per_class', type=int, default=3) 
    parser.add_argument('-num_classes', type=int, default=6)
    #parser.add_argument('-experiment_num', nargs=1, type=str, default='000')
    parser.add_argument('-run', type=str, default='run00')
    parser.add_argument('-losses', type=str, default='icnn')
    parser.add_argument('-pps_n_dim', type=int, default=128)
    # python main_crossV_Losses_ProtoPNet.py -dataset="./datasets/Mix_crossV/" -gpuid=0 \
	#				-base_architecture="resnet50" -da="OsDA" -pps_per_class=3 -num_classes=6 \
	#				-run="CrossV_run1" -losses="icnn"

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0] # [0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    global_vars.init()

    # book keeping namings and code
    from settings_icnn import img_size, prototype_activation_function, add_on_layers_type 
    base_architecture = args.base_architecture[0]
    pps_per_class= args.pps_per_class
    num_classes= args.num_classes
    pps_n_dim = args.pps_n_dim
    dataset_path = args.dataset
    dataset_name = dataset_path.split("/")[2] 
    Data_augmentation_mode = args.da
    #experiment_num = args.experiment_num[0]
    run = args.run#[0]
    print('--------------------------')     
    print("base_architecture", base_architecture, "type: ", type(base_architecture))
    print("pps_per_class", pps_per_class, "type: ", type(pps_per_class))
    print("num_classes", num_classes, "type: ", type(num_classes))
    
    #print("experiment_num", experiment_num, "type: ", type(experiment_num))
    print("run", run, "type: ", type(run))
    print("pps_n_dim", pps_n_dim, "type: ", type(pps_n_dim))

    prototype_shape = (num_classes*pps_per_class, pps_n_dim, 1, 1)
    print("prototype_shape", prototype_shape, "type: ", type(prototype_shape))
    print('--------------------------')

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    train_batch_size = 80 #128
    test_batch_size = 100 #128
    train_push_batch_size = 75 #128

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    from settings_icnn import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings_icnn import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings_icnn import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    from settings_icnn import coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    from settings_icnn import num_train_epochs, num_warm_epochs, push_start, push_epochs, fc_epochs
    #fc_epochs = 1 #20 

    # load the data
    # General images transformations (preprocess) for the training, Transforms for Push PPs and Validation are updated later
    if Data_augmentation_mode == "OsDA":
        dataset_transforms = transforms.Compose([
            #transforms.RandomChoice([
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.Pad(50, fill=0, padding_mode="symmetric"),
            #transforms.ColorJitter(brightness=0, contrast=[0.5, 1.5], saturation=0, hue=0),
            #transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5), 
            #transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            #transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2), scale=[0.5, 1]),
            #transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.5, hue=0),
            #transforms.RandomRotation(degrees=(-180, 180)),  
            #transforms.RandomAffine(degrees=(-45, 45)),
            #transforms.RandomAffine(degrees=(0,0), translate=(0.2, 0.2)),
            #transforms.RandomAffine(degrees=(0,0), scale=[0.8, 1]),                   
            #]),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Pad(50, fill=0, padding_mode="symmetric"),
                transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
                transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2), scale=[0.5, 1]),
                #transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.5, hue=0),
                transforms.RandomRotation(degrees=(-180, 180)),                     
            ]),            
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else: # This is the NoDA mode
        dataset_transforms = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    # Data transformations for the PPs
    push_transforms = transforms.Compose([           
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        #normalize,
    ])
    # Data transformations for the Validation-set
    val_transforms = transforms.Compose([   
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    #CrossV_dataset = datasets.ImageFolder(dataset_path) #, transform=dataset_transforms)
    CrossV_dataset = datasets.ImageFolder(dataset_path, transform=push_transforms)

    # Cross Validation definition, with K-fold=5: 
    k=5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    # train the model
    print('start training') #########################################################
    for fold, (train_idx, val_idx) in enumerate(kf.split(CrossV_dataset)):
        print(f"Fold {fold + 1}/{k}")
        train_subset = torch.utils.data.Subset(CrossV_dataset, train_idx)
        push_subset = torch.utils.data.Subset(CrossV_dataset, train_idx) #train_subset
        val_subset = torch.utils.data.Subset(CrossV_dataset, val_idx)

        # Update the transformations for the push subset
        train_subset.dataset.transform = dataset_transforms
        val_subset.dataset.transform = val_transforms
        push_subset.dataset.transform = push_transforms

        # Training set
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True,
            num_workers=4, pin_memory=False) #num_workers=4
        
        # Push set
        train_push_loader = torch.utils.data.DataLoader(
            push_subset, batch_size=train_push_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
        
        # Validation set
        test_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
        # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
        """ 
        print("Dataloaders: ")
        print(type(train_loader))
        print(type(train_push_loader))
        print(type(test_loader))
        # Iterate through DataLoader and print its contents
        for i, (data, target) in enumerate(train_push_loader):
            print(f"Batch {i+1}:")
            print("train_push_loader Data:")
            print(data)
            print("train_push_loader Target:")
            print(target)
            print("--------------------")
            break
        #"""

        model_dir = './saved_models/'+ run + '/' \
            + dataset_name + '_' + Data_augmentation_mode  + '_' +  base_architecture \
            +  '_' + str(pps_per_class) + 'pps_Fold' + str(fold) + '/'
        makedir(model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'settings_icnn.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

        log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
        img_dir = os.path.join(model_dir, 'img')
        makedir(img_dir)
        log('training set size: {0}'.format(len(train_loader.dataset)))
        log('push set size: {0}'.format(len(train_push_loader.dataset)))
        log('test set size: {0}'.format(len(test_loader.dataset)))
        log('batch size: {0}'.format(train_batch_size))
    
        prev_accu = 0.0
        prev_push_accu = 0.0
        prev_model_save_path = "./non-exitent_prev_model.pth"
        prev_push_model_save_path = "./non-exitent_prev_push_model.pth"
        for epoch in range(num_train_epochs):
            log('epoch: \t{0}'.format(epoch))

            if epoch < num_warm_epochs:
                # Update the transformations for the train subset
                train_subset.dataset.transform = dataset_transforms
                tnt.warm_only(model=ppnet_multi, log=log)
                _ = tnt.train(args, model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log) 
            else:
                # Update the transformations for the train subset
                train_subset.dataset.transform = dataset_transforms
                tnt.joint(model=ppnet_multi, log=log)
                joint_lr_scheduler.step()
                _ = tnt.train(args, model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
            # Update the transformations for the test subset
            val_subset.dataset.transform = val_transforms
            accu = tnt.test(args, model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            if (prev_accu < accu):
                prev_accu = accu
                delete_previous_models(prev_model_save_path)
                prev_model_save_path = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush_', accu=accu)#, target_accu=0.0, log=log)
                print(f"saved model at:   {prev_model_save_path}")

            if epoch >= push_start and epoch in push_epochs:
                # Update the transformations for the push subset
                push_subset.dataset.transform = push_transforms
                print("previous to push: ", type(train_push_loader))
                print(train_push_loader)
                push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=preprocess_input_function, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log)
                # Update the transformations for the test subset
                val_subset.dataset.transform = val_transforms
                accu = tnt.test(args, model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                if (prev_push_accu < accu):
                    prev_push_accu = accu       
                    delete_previous_models(prev_push_model_save_path)                     
                    prev_push_model_save_path = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push_', accu=accu)#, target_accu=0.0, log=log)

                if prototype_activation_function != 'linear':
                    # Update the transformations for the train subset
                    train_subset.dataset.transform = dataset_transforms
                    tnt.last_only(model=ppnet_multi, log=log)
                    for i in range(fc_epochs):
                        model_was_saved = 0
                        print('epoch: \t{0}'.format(epoch))
                        log('iteration: \t{0}'.format(i))
                        _ = tnt.train(args, model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                    class_specific=class_specific, coefs=coefs, log=log)
                        
                        # Update the transformations for the test subset
                        val_subset.dataset.transform = val_transforms
                        accu = tnt.test(args, model=ppnet_multi, dataloader=test_loader,
                                        class_specific=class_specific, log=log)
                        if (prev_push_accu < accu):
                            prev_push_accu = accu
                            delete_previous_models(prev_push_model_save_path)
                            prev_push_model_save_path = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push_', accu=accu)#, target_accu=0.0, log=log)
                            model_was_saved = 1
                        #if (i == (fc_epochs - 1)) and (model_was_saved == 0):
                        #     _ = save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push_', accu=accu)#, target_accu=0.0, log=log)
        
        fold_accuracies.append(prev_push_accu)
        print('--------------------------')
        print(model_dir)
        print("Best accuracy: ", prev_push_accu)
        print('--------------------------')
        logclose()

    print(f"Average Validation Loss: {sum(fold_accuracies) / len(fold_accuracies):.4f}")

    csv_accuracies_path = './saved_models/'+ run + '/' 
    with open(csv_accuracies_path + 'accuracies.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for accuracy in fold_accuracies:
            writer.writerow([accuracy])
