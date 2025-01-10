# import module
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# import file's function or class
from argument import args_parser
from dataset import get_dataset
from model import SemanticCommunicationSystem
from local_model import LocalUpdate


if __name__ == '__main__':

    # SummaryWriter : create an event file in a given directory and add summaries and events to it
    # log file / event file : typically used by software or operating systems to keep track of certain events that occur
    logger = SummaryWriter('../logs')

    args = args_parser()

    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'
    print('Using ' + device + '!!!\n')

    # Load dataset
    print('Loading ' + args.dataset + ' dataset ...')
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

    for epoch in range(args.epochs): 
        local_weights, local_losses = [], []

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
        # a: 0～a的整數區間或其他的陣列資料, size: 輸出的陣列大小, replace: 隨機數是否重複(預設True), p: 陣列資料的機率分布(加總為 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print('\nTotal number of edge devices are selected : '+ str(len(idxs_users)))
        print('Selected edge devices number : '+ str(idxs_users))

        for idx in idxs_users: 
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)