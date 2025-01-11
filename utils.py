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
    if args.gpu : 
        print(f'    Device : GPU')
    else : 
        print(f'    Device : CPU')
    print(f'    Dataset : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning rate  : {args.lr}')

    print('    Federated parameters:')
    print(f'    Global Rounds   : {args.global_ep}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Number of users   : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}\n')
    
    
    return

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    torch.cuda.empty_cache()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    total_count, correct_count = 0.0, 0.0

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
        
        correct += torch.sum(torch.eq(s_predicted, s_origin)).item()
        total += len(s_origin)
        """
        # 假設 s_predicted 和 s_origin 已經從模型輸出
        # s_predicted: [2, 3, 32, 35632]
        # s_origin: [2, 3, 32, 32]

        # 提取預測類別
        predicted_labels = torch.argmax(s_predicted, dim=1)  # [2, 32, 35632]

        # 提取真實類別
        true_labels = torch.argmax(images, dim=1)  # 假設 s_origin 是 one-hot，形狀為 [2, 32, 32]

        # 匹配形狀
        predicted_labels = predicted_labels[:, :, :true_labels.size(2)]  # 確保形狀一致

        # 計算準確率
        correct = torch.eq(predicted_labels, true_labels)  # [2, 32, 32] 的布林值張量
        correct_count += correct.sum().item()
        total_count += correct.numel()
        accuracy = correct_count / total_count
        print(f"Accuracy: {accuracy:.2%}")

    accuracy = correct_count/total_count
    print(accuracy)
    return accuracy, loss
    



