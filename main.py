# import module
import torch
import numpy as np
import copy
import pickle
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# import file's function or class
from argument import args_parser
from dataset import get_dataset
from model import SemanticCommunicationSystem
from local_model import LocalUpdate
from utils import FedAvg, FedLol, exp_details, test_inference


if __name__ == '__main__':
    # use to measure total running time
    start_time = time.time()

    # SummaryWriter : create an event file in a given directory and add summaries and events to it
    # log file / event file : typically used by software or operating systems to keep track of certain events that occur
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # Load dataset
    print('Loading ' + args.dataset + ' dataset ...\n')
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Semantic Communication model
    N_channels = 256
    Kernel_sz = 5
    IMG_SIZE = [3, 32, 32]
    enc_out_shape = [2, IMG_SIZE[1]//4, IMG_SIZE[2]//4]
    global_SCmodel = SemanticCommunicationSystem(enc_out_shape, Kernel_sz, N_channels).cuda()
    # Set the model to train and send it to device(cpu or gpu).
    global_SCmodel.to(device)
    # the model is in training mode, enable batch normalization and dropout
    global_SCmodel.train()

    #print(global_SCmodel)
    # copy weights
    # state_dict() : a Python dictionary object that maps each layer to its parameter tensor.
    global_weights = global_SCmodel.state_dict()
    #print(global_weights)
    
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre = 0
    curr_user = 0   # 目前第幾個edge device在跑

    for epoch in range(args.global_ep): 
        local_weights, local_losses = [], []
        curr_user = 0

        global_SCmodel.train()

        """
        in ref. paper : "Communication-Efficient Learning of Deep Networks from Decentralized Data" 中提到
        At the beginning of each round, a random fraction of clients is selected, and the server 
        sends the current global algorithm state to each of these clients (e.g., the current model parameters). 
        We only select a fraction of clients for efficiency
        計算 m = frac*user, 從所有的user中random挑出m個user
        """
        # defaulted m : 0.1 * 100 = 10
        m = max(int(args.frac * args.num_users), 1)
        # numpy.random.choice(a, size=None, replace=True, p=None): 會產生指定數量的一維陣列隨機整數
        # a: 0～a的整數區間或其他的陣列資料, size: 輸出的陣列大小, 
        # replace: 隨機數是否重複(預設True), p: 陣列資料的機率分布(加總為 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #print('\nTotal number of edge devices are selected : '+ str(len(idxs_users)))
        #print('Selected edge devices number : '+ str(idxs_users))
        # local model training
        for idx in idxs_users: 
            curr_user = curr_user + 1
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)

            # copy.deepcopy() : 完全複製了一份副本，兩者指向的memory address不一樣，
            #                   Modify the deep copied object do not affect the original object
            # copy.copy() : shallow copy, If you modify a the shallow copied object, 
            #               the change will reflect in the original object because they share the same reference
            w, loss = local_model.update_weights(curr_user, idx, model=copy.deepcopy(global_SCmodel), global_round=epoch)
            # 把deepcopy()改成用 '=', global model和model的addr會相同，改的時候會一起改

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weight strategy
        # Federated Average
        #global_weights = FedAvg(local_weights)
        # Federated Local Loss
        global_weights = FedLol(local_weights, local_losses)

        # update global model
        global_SCmodel.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        # eval() : do not enable batch normalization and dropout
        # 在輸入時若不做訓練仍然會改變權重, 這是因為model中有BN和dropout layer, eval()時會把BN跟dropout固定住
        global_SCmodel.eval()
        
        print('\nGlobal Round: {} - Local model inference ...'.format(epoch+1))
        for c in tqdm(range(args.num_users), colour="yellow"):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(c, model=global_SCmodel)
            list_acc.append(acc)
            list_loss.append(loss)
            
        train_accuracy.append(sum(list_acc)/len(list_acc))
        # print global training loss after every 'i' rounds
        #if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[0]))
    
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_SCmodel, test_dataset)

    print(f' \n Results after {args.global_ep} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[0]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.global_ep, args.frac, args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING 畫圖
    import matplotlib.pyplot as plt

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.global_ep, args.frac, args.local_ep, args.local_bs))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_B[{}]_acc.png'.
                format(args.dataset, args.global_ep, args.frac, args.local_ep, args.local_bs))