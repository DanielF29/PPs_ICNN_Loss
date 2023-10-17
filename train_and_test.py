import time
import torch
import torch.nn as nn

from helpers import list_of_distances, make_one_hot
from algorithms.icnn_loss_pps_2 import get_pps_icnn_loss
import utils.globals as global_vars

def _train_or_test(args, model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    losses = args.losses
    condition_PPs_loss = False
    condition_icnn_loss = False
    condition_Gaffinity_loss = False
    condition_PPs_plus_ICNN_loss = False

    if "pps_loss" in losses:
        condition_PPs_loss = True
    if "icnn_loss" in losses:
        condition_icnn_loss = True
    if "Gaffinity" in losses:
        condition_Gaffinity_loss = True
    if "pps_plus_icnn" in losses:
        condition_PPs_plus_ICNN_loss = True

    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        #print("0 -- image.size(): ", image.size(), " Max: ", torch.max(image).item(), " Min: ", torch.min(image).item() ) 
        input = image.cuda()
        target = label.cuda()
        
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, conv_features = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if condition_Gaffinity_loss:
                pass # Pending to be implemented

            if condition_icnn_loss:
                # Code toi compute the ICNN_score of a batch of inputs and the Prototypical parts (PPs)
                num_classes = args.num_classes#[0]
                pps_per_class = args.pps_per_class#[0]
                pps_fts = model.module.prototype_vectors.view(model.module.num_prototypes, -1)#.cpu()
                input_fts = conv_features #global_vars.imgs_fts # model.module.conv_features(input) 
                # Flatten the features with avg pooling
                pps_n_dim = args.pps_n_dim#[0]
                imgs_embeding_size_output = input_fts.shape[-1]
                batch_size = input_fts.shape[0] # [input_fts.shape[0], input_fts.shape[1]] 
                input_fts = nn.AvgPool2d(imgs_embeding_size_output)(input_fts).reshape(batch_size, -1)
                #    # The previous line, reduces the imgs in the batch embedings from torch.Size([80, 128, 7, 7])
                #    # to torch.Size([80, 128])
                #print("pps_fts type: ", type(pps_fts), "input_fts type: ", type(input_fts))
                #print("pps_fts shape: ", pps_fts.shape, "input_fts shape: ", input_fts.shape, "input_labels shape: ", target.shape)
                icnn_pps_loss = get_pps_icnn_loss(pps_fts, input_fts, num_classes, pps_per_class, input_labels=target)
                #print("icnn_pps_loss: ", icnn_pps_loss)

            if condition_PPs_loss:
                #"""
                if class_specific:
                    max_dist = (model.module.prototype_shape[1]
                                * model.module.prototype_shape[2]
                                * model.module.prototype_shape[3])

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    # calculate avg cluster cost
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                    
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.module.last_layer.weight.norm(p=1) 

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.module.last_layer.weight.norm(p=1)
                #""" 
            
            if condition_PPs_plus_ICNN_loss:
                # Code to compute the ICNN_score of a batch of inputs and the Prototypical parts (PPs)
                num_classes = args.num_classes#[0]
                pps_per_class = args.pps_per_class#[0]
                pps_fts = model.module.prototype_vectors.view(model.module.num_prototypes, -1)#.cpu()
                input_fts = conv_features #global_vars.imgs_fts # model.module.conv_features(input) 
                # Flatten the features with avg pooling
                pps_n_dim = args.pps_n_dim#[0]
                imgs_embeding_size_output = input_fts.shape[-1]
                batch_size = input_fts.shape[0] # [input_fts.shape[0], input_fts.shape[1]] 
                input_fts = nn.AvgPool2d(imgs_embeding_size_output)(input_fts).reshape(batch_size, -1)
                #    # The previous line, reduces the imgs in the batch embedings from torch.Size([80, 128, 7, 7])
                #    # to torch.Size([80, 128])
                #print("pps_fts type: ", type(pps_fts), "input_fts type: ", type(input_fts))
                #print("pps_fts shape: ", pps_fts.shape, "input_fts shape: ", input_fts.shape, "input_labels shape: ", target.shape)
                icnn_pps_loss = get_pps_icnn_loss(pps_fts, input_fts, num_classes, pps_per_class, input_labels=target)
                #print("icnn_pps_loss: ", icnn_pps_loss)

                # Code to compute the PPs loss
                #"""
                if class_specific:
                    max_dist = (model.module.prototype_shape[1]
                                * model.module.prototype_shape[2]
                                * model.module.prototype_shape[3])

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    # calculate avg cluster cost
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                    
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.module.last_layer.weight.norm(p=1) 

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.module.last_layer.weight.norm(p=1)
                #"""                            

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            #total_cluster_cost += cluster_cost.item()
            #total_separation_cost += separation_cost.item()
            #total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if condition_PPs_loss:
                #"""
                if class_specific:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
                else:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #"""

            if condition_icnn_loss:
                loss = cross_entropy + icnn_pps_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if condition_Gaffinity_loss:
                pass # Pending to be implemented

            if condition_PPs_plus_ICNN_loss:
                #"""
                if class_specific:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1
                            + coefs['icnn'] * icnn_pps_loss)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 1.0*icnn_pps_loss
                else:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1
                            + coefs['icnn'] * icnn_pps_loss)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1 + 1.0*icnn_pps_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #"""


        del input
        del target
        del output
        del predicted
        del min_distances
    
    #print('batch accu: {0:.3f}%'.format(n_correct / n_examples * 100), flush=True)
    end = time.time()
    
    #(f'TRAIN | Epoch {epoch} | Loss={loss.item():.4f} | Acc={acc*100:.2f} |')
    log('\ttime: \t{0:.3f}'.format(end -  start))
    #log('\tcross ent: \t{0:.4f}'.format(total_cross_entropy / n_batches))
    #log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    #if class_specific:
    #    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    #    log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0:.3f}%'.format(n_correct / n_examples * 100))
    #log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(args, model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(args, model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(args, model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(args, model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
