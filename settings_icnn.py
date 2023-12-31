#base_architecture = 'resnet152'
img_size = 224
#num_classes = 6
#PPs_per_class = 10
#prototype_shape = (num_classes*PPs_per_class, 128, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

#experiment_run = '001'
#---------------------------
# load the data
data_path =   './datasets/Mix/' #UAMS_Plus_Nachos_Balanced/' #'./datasets/tiny-imagenet-200/' #'./datasets/cub200_cropped/'
train_dir = data_path + 'train/' #'train_augmented/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75
#---------------------------
# define optimizer
joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4
#---------------------------
# weighting of different training losses
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'icnn': 1,
}
#---------------------------
num_train_epochs = 31 #31
num_warm_epochs = 5 #5

push_start = 10  #10 # The initial epoch in which push of PPs begings
push_period = 10 # 10 # The number of cnn epochs between each push of PPs 
push_epochs = [i for i in range(num_train_epochs) if i % push_period == 0]

fc_epochs = 20 # 20
#---------------------------
#---------------------------