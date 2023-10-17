# ICNN: Improved Clustering-based Nearest Neighbor Search
# Modified by: Mauricio Mendez-Ruiz
# ICNN paper: https://arxiv.org/pdf/2107.06992.pdf
# Original ICNN score by: Ivan Alejandro García Ramírez
# Original ICNN work: Supervised feature space reduction for multi-label classification

import torch
import torch.nn.functional as F

import util.globals as glob_var
def init():
    global supp_fts
    global query_fts
    global prototypes

#   modified_score(features, labels, ipc=ipc)
#                                    ipc = args.shot+args.query
def modified_score(origin, y_orig, target=None, y_targ=None, k=4, ipc=None):
    target = origin if type(target) == type(None) else target
    y_targ = y_orig if type(y_targ) == type(None) else y_targ
    k = target.shape[0] - 2 if k+2 >= target.shape[0] else k
    ipc = k+1 if type(ipc) == type(None) else ipc
    eps = 0.0000001

    distances, indices = nearest_neighbors(origin, target, k=ipc+1) 
    # Using the def: topk_lowest_without_diag given below the next 2 lines are not needed
    distances = distances[:,1:] # shape after this: 30x10. 30 rows and 10 columns for a 5 way, 5 shot, Queries=5. 30= 5x5+5. The first column was removed.
    indices = indices[:,1:]

    # class by neighbor
    classes = y_targ[indices] # final shape (30,10)
    yMatrix = y_orig.repeat(ipc,1).T
    scMatrix = (yMatrix == classes)*1 # Same class matrix [1 same class, 0 diff class]
    dcMatrix = (scMatrix)*(-1)+1 # Different class matrix [negation of scMatrix]

    ### Lambda Computation ###

    # Same class distance
    dt = distances.T # shape 10x30
    dcnd = distances*dcMatrix # lowest distances to samples that belong to a different class 
                              #  from each of the supp*way + query samples that are part of the batch. Tensor of shape (30,10)
    nn_dc = (dcnd + torch.where(dcnd.eq(0.), float('inf'), 0.)).min(axis=1).values # Final tensor of shape (30,)
    nd = dt / (nn_dc + eps) # the distances are Standarize with respect to the shortes distance of a sample to the closest one of a different class. 
                            # Tensor shape (10,30)
    nd = nd[:k, :] # Only the Standarized distances of the same class are expected to remain with respect to each sample in the batch. Tensor Shape (4,30)
    nd = nd / torch.stack((torch.ones_like(nd[0]), nd.max(axis=0).values)).max(axis=0).values # Normalize with max(max_from_row, 1.0)
            # in the previous line: values of nd are re-standarized to be between 0 and 1 per column. Tensor shape (10,30)
    nd = nd.T

    scMatrix = scMatrix[:, :k] # Ideally we kept only the section of the scMatrix that has 1's 
    scnd = nd*scMatrix # Here we asure we kept only results of Standarized distances of the same class or a "cero"

    scCounts = scMatrix.sum(axis=1) # we get the number of samples that are within the "k" closest that are of the same class as the corresponding sample per row
    scndsum = scnd.sum(axis=1) / (scCounts + eps) # average of standarized distances per row, corresponing to one sample
    sclamb = 1 - (scndsum.sum() / (torch.count_nonzero(scCounts) + eps)) # This is 1, minus the average of averages of the distances intra-class (within a class)
    # torch.count_nonzero(scCounts) returns the total number of elements in the tensor that have a non-zero value

    dcnd = dcnd[:, :k]
    dcMatrix = dcMatrix[:, :k]

    # Different class distance
    dcnd = dcnd / (dcnd.max() + eps) # Estandarized distances per row, againt the closest "k" samples.
    dcop = -1 if torch.all(dcMatrix == 0) else 0 
    dcCounts = dcMatrix.sum(axis=1) # count per row of the number of instances of different class within the closest "k" samples
    dcndsum = dcnd.sum(axis=1) / (dcCounts + eps) # Standarized Distance average per row 
    dclamb = dcndsum.sum() / (torch.count_nonzero(dcCounts) + eps) # This is 1, minus the average of averages of the distances intra-class (within a class)
    # torch.count_nonzero(dcCounts) returns the total number of elements in the tensor that have a non-zero value
    dclamb = torch.abs(dclamb + dcop)

    lambr = (sclamb + dclamb) / 2

    ## Omega Calculation ###
    # varsc = torch.var(scnd)
    # vardf = torch.var(dcnd)
    # omega = 1 - (varsc+vardf)
    
    ### Gamma computation
    gamma = torch.sum(torch.sum(scMatrix, axis=1) / k) / (y_orig.shape[0]) if k+2 < target.shape[0] else 1.0
    
    # return (lambr + gamma + omega) / 3
    return (lambr + gamma) / 2

def distance_matrix(x, y=None, p=2):
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, p).sum(2)
    
    return dist

def nearest_neighbors(X, y=None, k=3, p=2):
    eps = 0.000001
    dist = (distance_matrix(X, y, p=p) + eps) ** (1/2)
    knn = dist.topk(k, largest=False)
    #values, indices = topk_lowest_without_diag(dist, k)
    return knn.values, knn.indices #values, indices #knn.values, knn.indices

def proto_triplet_loss(queries, y_qry, protos, y_protos, way=5, margin=0.5):
    distances, indices = nearest_neighbors(queries, protos, k=way)
    classes = y_protos[indices]
    yMatrix = y_qry.repeat(way,1).T
    scMatrix = (yMatrix == classes)*1
    dcMatrix = (scMatrix)*(-1)+1

    scd = distances*scMatrix
    scp = scd.max(axis=1).values

    dcd = distances*dcMatrix
    dcd += torch.where(dcd.eq(0.), float('inf'), 0.)
    dcp = dcd.min(axis=1).values


    dists = torch.stack((scp - dcp + margin, torch.zeros_like(scp)), axis=1)
    return torch.mean(torch.max(dists, axis=1).values)

# Not in use currently because the interesting lowest distances 
#   are expected always at the right size of the diagonal of the dist matrix
def topk_lowest_without_diag(dist, k):
    # Create a copy of the dist tensor
    dist_copy = dist.clone()
    
    # Set the diagonal of the dist_copy tensor to float('inf')
    dist_copy.fill_diagonal_(float('inf'))
    
    # Get the top-k lowest values and their indices from the dist_copy tensor
    values, indices = dist_copy.topk(k, largest=False)
    
    return values, indices

# loss = get_icnn_loss(args, logits, args.train_way, train_label)    
def get_icnn_loss(args, logits, way, qry_labels):
    loss = 0

    if 'cross' in args.losses:
        loss = F.cross_entropy(logits, qry_labels)

    if 'suppicnn' in args.losses:
        supp_labels = torch.arange(0, way, 1/args.shot).type(torch.int).cuda()
        supp_score = modified_score(glob_var.supp_fts, supp_labels)
        loss += (-torch.log(supp_score))

    if 'queryicnn' in args.losses:
        if args.query_protos:
            proto_labels = torch.arange(0, way).type(torch.int).cuda()
            query_score = modified_score(glob_var.query_fts, qry_labels, glob_var.prototypes, proto_labels)
        else:
            supp_labels = torch.arange(0, way, 1/args.shot).type(torch.int).cuda()
            query_score = modified_score(glob_var.query_fts, qry_labels, glob_var.supp_fts, supp_labels)
        loss += (-torch.log(query_score))

    if 'fullicnn' in args.losses:
        features = torch.cat((glob_var.supp_fts, glob_var.query_fts))
        supp_labels = torch.arange(0, way, 1/args.shot).type(torch.int).cuda()
        labels = torch.cat((supp_labels, qry_labels))
        ipc = args.shot+args.query
        score = modified_score(features, labels, ipc=ipc)

        loss += (-torch.log(score))
    
    if 'prototriplet' in args.losses:
        proto_labels = torch.arange(0, way).type(torch.int).cuda()
        loss += proto_triplet_loss(glob_var.query_fts, qry_labels, glob_var.prototypes, proto_labels, way)

    count = len(args.losses.split(','))
    loss = loss / count

    return loss

def score(origin, y_orig, target=None, y_targ=None, k=5, p=2, q=2, r=2):
    """Compute class prototypes from support samples.

    # Arguments
        X: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        y: torch.Tensor. The class of every sample
        k: int. the number of neigbhors (small k focus on local structures big k on global)
        p: int. must be a natural number, the higher is p, the lower penalization on lambda function
        q: int. must be a natural number, the higher is p, the lower penalization on omega function
        r: int. must be a natural number, the higher is r, the lower penalization on gamma function

    # Returns
        
    """
    target = origin if type(target) == type(None) else target
    y_targ = y_orig if type(y_targ) == type(None) else y_targ

    eps = 0.000001
    k = 3 if k >= target.shape[0] else k

    #min max scale by feature
    # a = target - target.min(axis=0).values
    # b = X.max(axis=0).values - X.min(axis=0).values
    # X = torch.divide(a , b+eps)

    distances, indices = nearest_neighbors(origin, target, k=k+1)
    distances = distances[:,1:]
    indices = indices[:,1:]

    # class by neighbor
    classes = y_targ[indices]
    yMatrix = y_orig.repeat(k,1).T
    scMatrix = (yMatrix == classes)*1 # Same class matrix [1 same class, 0 diff class]
    dcMatrix = (scMatrix)*(-1)+1 # Different class matrix [negation of scMatrix]

    ### Normalizing distances between neighbords
    dt = distances.T
    nd = (dt - dt.min(axis=0).values) / ( (dt.max(axis=0).values - dt.min(axis=0).values) + eps )
    nd = nd.T

    ## Distances
    scd = distances*scMatrix #Same class distance
    dcd = distances*dcMatrix #Different class distance
    ## Normalized distances
    scnd = nd*scMatrix #Same class normalized distance
    dcnd = nd*dcMatrix #Different class normalized distance
    
    ### Lambda computation
    plamb = (1 - scnd) * scMatrix
    lamb = (dcnd + plamb)
    lambs = torch.sum(lamb,axis=1)
    lambs2 = (lambs / (torch.max(lambs) + eps)) ** (1/p)
    lambr = torch.sum(lambs2) / (y_orig.shape[0])

    varsc = torch.var(scnd)
    vardf = torch.var(dcnd)
    omega = (1 - (varsc+vardf))**(1/q)
    
    gamma = torch.sum((torch.sum(scMatrix, axis=1) / k) ** (1/r)) / (y_orig.shape[0])
    
    # return (lambr + gamma + omega)/3
    # return lambr
    return (lambr + gamma)/2