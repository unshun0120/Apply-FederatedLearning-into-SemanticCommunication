import argparse

# input arguments to set the parameters value
def args_parser():
    parser = argparse.ArgumentParser()

    # environment setup
    # GPU or CPU
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    # verbose, 是否輸出進度條、loss、acc
    parser.add_argument('--verbose', type=int, default=1, help='whether output progress bar, loss value, accuracy when training')


    # model something setup
    # Dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="name of dataset. Default set to use CIFAR10")
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help="type of optimizer")
    # Learning rate
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    


    # Federated Learning setup 
    # Epochs
    parser.add_argument('--epochs', type=int, default=1, help="number of training rounds")
    # number of edge devices (i.e. users, clients)
    parser.add_argument('--num_users', type=int, default=100, help="number of edge devices (users): K")
    # Fraction of selected edge devices
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    # Local model epochs
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    # Local model batch size
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")




    args = parser.parse_args()
    return args

