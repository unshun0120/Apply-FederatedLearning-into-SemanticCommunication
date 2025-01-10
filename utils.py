import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def FedAvg(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    # k (string) = conv1.weight, conv1.bias, conv2.weight...
    # i = i-th edge device
    # 把第k個layer中每個device的weight加起來
    for k in w_avg.keys():  
        for i in range(1, len(w)):  
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg

def FedLol(w, l):
    """
    Returns the Federated Local Loss.
    """
    w_lol = copy.deepcopy(w[0])
    l_lol = copy.deepcopy(l)
    # compute sum of loss
    l_sum = float(0)
    for L in range(1, len(l_lol)) : 
        l_sum += l_lol[L]
    # compute local loss weight
    for i in range(1, len(l_lol)) : 
        l_lol[i] = ((l_sum-l_lol[i])/l_sum) / (len(l_lol)-1)
    # allocate local loss to each layer weight
    for k in w_lol.keys():  
        for i in range(1, len(w_lol)):  
            w_lol[k] += w[i][k] * l_lol[i]
    
    return w_lol

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        CHANNEL = 'AWGN' 
        SNR_TRAIN = torch.randint(0, 28, (images.shape[0], 1)).cuda()
        CR = 0.1+0.9*torch.rand(images.shape[0], 1).cuda()
        s_predicted, s_origin= model(images, SNR_TRAIN, CR, CHANNEL)

        # 計算loss時, predicted和origin的shape要相同, 用填充(padding)的方式讓s_origin和s_predicted相同
        padding = (0, s_predicted.shape[3] - s_origin.shape[3])  # 只在最後一維填充
        s_origin = F.pad(s_origin, padding)

        batch_loss = criterion(s_predicted, s_origin)
        loss += batch_loss.item()

        # Prediction
        """ 
        _, pred_labels = torch.max(s_predicted, 1)
        pred_labels = pred_labels.view(-1)
        """ 
        correct += torch.sum(torch.eq(s_predicted, s_origin)).item()
        total += len(s_origin)

    accuracy = correct/total
    return accuracy, loss