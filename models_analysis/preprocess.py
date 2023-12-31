import torch

# Preprocess parameters for JE Kidney Stones dataset
mean = (0.4864,	0.5720,	0.6475) # (0.485, 0.456, 0.406) #original mean from CUB?
std =  (0.1533,	0.1558,	0.1587) # (0.229, 0.224, 0.225) #original std from CUB?

# Preprocess parameters for UAMS mammograms dataset
#mean = [0.1860, 0.2479, 0.2642]
#std = [0.2407, 0.3060, 0.3085]

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)
