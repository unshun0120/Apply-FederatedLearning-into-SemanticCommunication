import torch.nn
import torch.nn as nn

# The Encoder model with attention feature blocks
class Encoder(nn.Module):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()    
        self.encoder = nn.Sequential( 
            # semantic encoder通常使用 CNN 或 Swin Transformer 提取圖像語義特徵

            # torch.nn.Conv2d(): 卷積, 卷積是把A變成B, 反卷積則是把B變成A
            # 3: 輸入channel數，CIFAR10的圖片是RGB，所以channel數=3
            # 64: 輸出的channel數，也就是卷積過後的特徵圖數量，這裡使用了64個filter來捕捉不同的特徵
            # kernel_size=3: filter通常使用的大小是3×3，所以kernel size=3，能夠有效捕捉小範圍的特徵
            # stride=1: 卷積的步長是 1，表示每次卷積操作會平移 1 個像素，從而保持影像的解析度 
            # padding=1: 填充（padding）是 1，這樣做可以保證輸入和輸出的尺寸相同，避免尺寸縮小。在 3x3 的濾波器下，padding=1 使得卷積後的特徵圖大小保持不變（32x32）
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            # Batch Normalization=64: 前面卷積層輸出的channel數 = 64
            # BN: 作用是加速訓練並提高穩定性，通過正規化每個batch的特徵圖來減少內部偏移，從而使得訓練過程更加穩定和高效
            nn.BatchNorm2d(64), 
            # ReLU(x) = max(0, x)，將負數設0
            nn.ReLU(), 
            # MaxPool2d : 最大池化（MaxPooling），作用是對特徵圖進行降維操作，從而減少計算量並保留最重要的特徵
            # kernel_size=2：池化核的大小是 2x2，意味著每次池化操作會選擇 2x2 區域內的最大值
            # stride=2：池化的步長是 2，這表示每次池化操作會跳過 2 個像素，從而使得特徵圖的大小減少一半
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        ) 
        # Fully connected layer to project features to the semantic space
        self.fc = nn.Linear(4096, output_dim)

    def forward(self, x):    
        x = self.encoder(x)
        # x.view(x.size(0), -1): 是為了將輸出的tensor轉為一個一維向量。這樣做是為了便於後續的全連接層（ChannelEncoder）的操作，因為全連接層一般需要輸入一維向量而不是多維張量
        # x.size(0): 是batch的大小，保持不變。
        # -1: 表示自動計算這個維度的大小以確保總元素數量不變
        # e.g. 如果 x 的大小是 [64, 256, 4, 4]（即batch大小為64，每張圖像經過卷積和池化後的通道數為256，空間尺寸為4x4），
        # 那麼 x.view(x.size(0), -1) 會將其轉換為 [64, 4096]，其中4096是25644的結果
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

def dense(input_size, output_size):  # using dense layer : dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU()
    )

# The (real) AWGN channel    
def AWGN_channel(x, snr, P = 1):  
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(P/gamma)*torch.randn(batch_size, length).cuda()
    y = x+noise
    return y

class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 8 * 8 * 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # semantic decoder通常使用反卷積或 U-Net 結構還原圖像
            # torch.nn.ConvTranspose2d(): 反卷積, 卷積是把A變成B, 反卷積則是把B變成A
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 8, 8)
        x = self.decoder(x)
        return x


  