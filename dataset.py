from torchvision import datasets, transforms
import numpy as np

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'CIFAR10':
        data_dir = '../Dataset/CIFAR10/'
        # Composes several transforms together
        # ToTensor() : 將shape為(H, W, C)的numpy array or image轉成(C, H, W)的tensor，並把每個值normalize到[0, 1], normalize方式 : 把每個值除以255
        # Normalize() : 將輸入normalize到[0, 1]，再用公式(x-mean)/std，把每個元素分布到[-1, 1]
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(std=(0.5, 0.5, 0.5),mean=(0.5, 0.5, 0.5))])
        # dataset.CIFAR10(root, train, transform, download) 
        # root (data_dir) : "where directory cifar-10 exists or will be saved to", 
        # train : "If True, creates dataset from training set, otherwise creates from test set", 
        # download : "If true, downloads the dataset from the internet and puts it in root directory"
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
        num_items = int(len(train_dataset)/args.num_users)
        user_groups, all_data_idxs = {}, [i for i in range(len(train_dataset))]
        for i in range(args.num_users):
            user_groups[i] = set(np.random.choice(all_data_idxs, num_items,
                                                replace=False))
            all_data_idxs = list(set(all_data_idxs) - user_groups[i])

    elif args.dataset == 'MNIST':
        data_dir = '../Dataset/MNIST/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
    return train_dataset, test_dataset, user_groups