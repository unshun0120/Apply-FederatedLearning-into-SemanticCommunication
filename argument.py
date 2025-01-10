import argparse

# input arguments to set the parameters value
def args_parser():
    parser = argparse.ArgumentParser()

    # environment setup
        # GPU or CPU
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")

    # setup
        # Dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="name of dataset. Default set to use CIFAR10")






    args = parser.parse_args()
    return args

