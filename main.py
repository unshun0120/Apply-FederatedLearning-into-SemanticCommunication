# import module
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# import file's function or class
from argument import args_parser
from dataset import get_dataset
from model import SemanticCommunicationSystem





if __name__ == '__main__':
    args = args_parser()

    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'
    print('Using ' + device + '!!!')

    # Load dataset
    print('Loading ' + args.dataset + ' dataset ...')
    train_dataset, test_dataset = get_dataset(args)

    # Semantic Communication model
    N_channels = 256
    Kernel_sz = 5
    IMG_SIZE = [3, 32, 32]
    enc_out_shape = [2, IMG_SIZE[1]//4, IMG_SIZE[2]//4]
    net = SemanticCommunicationSystem(enc_out_shape, Kernel_sz, N_channels).cuda()