import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)
    # 只要call DatasetSplit()就會去call __getitem__(), e.g. idxs_train是list, idx=0時會call DatasetSplit然後return dataset[0]的三個東西(作者function define), 依此類推
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return torch.as_tensor(image), torch.as_tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.MSELoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80%, 10%, 10%)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=False)
        
        return trainloader, validloader, testloader

    def update_weights(self, curr_user, user_idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        print('\nGlobal Round: {}, {}-th Edge device, Local model ID: {}'.format(global_round+1, curr_user, user_idx))
        for local_epoch in range(self.args.local_ep):
            batch_loss = []
            # _: batch size
            for _, (images, labels) in enumerate(tqdm(self.trainloader, desc="Local Round {} ...".format(local_epoch+1))):
                images, labels = images.to(self.device), labels.to(self.device)

                # model.zero_grad() and optimizer.zero_grad() are the same if all model parameters are in that optimizer
                model.zero_grad()

                
                SNR_TRAIN = torch.randint(0, 28, (images.shape[0], 1)).cuda()
                s_predicted, s_origin= model(images, SNR_TRAIN)
                # 計算loss時, predicted和origin的shape要相同, 用填充(padding)的方式讓s_origin和s_predicted相同
                padding = (0, s_predicted.shape[3] - s_origin.shape[3])  # 只在最後一維填充
                s_origin = F.pad(s_origin, padding)
                # get loss value
                loss = self.criterion(s_predicted, s_origin)
                loss.backward()
                optimizer.step()
                
                # verbose : 是否要顯示進度條, 預設1
                # verbose = 0 : 不輸出進度條、loss、acc, verbose = 1 : 輸出進度條、loss、acc, verbose = 2 : 輸出loss、acc但不輸出進度條
                """
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | {}-th Local Model {} Epoch : {} | Batch Index : {} | [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                        global_round, curr_user, user_idx, local_epoch, batch_idx, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                """
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, idx, model):
        """ 
        Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        total_count, correct_count = 0.0, 0.0

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # Inference
                SNR_TRAIN = torch.randint(0, 28, (images.shape[0], 1)).cuda()
                s_predicted, s_origin= model(images, SNR_TRAIN)

                # 計算loss時, predicted和origin的shape要相同, 用填充(padding)的方式讓s_origin和s_predicted相同
                padding = (0, s_predicted.shape[3] - images.shape[3])  # 只在最後一維填充
                images = F.pad(images, padding)

                batch_loss = self.criterion(s_predicted, images)
                loss += batch_loss.item()

                # Prediction
                """
                _, pred_labels = torch.max(s_predicted, 0)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                """
                # 假設 s_predicted, s_origin樣子是
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
                #accuracy = correct_count / total_count
                #print(f"idx {idx} -> Accuracy: {accuracy:.2%}")

        accuracy = correct_count/total_count
        #print(accuracy)
        return accuracy, loss
    
