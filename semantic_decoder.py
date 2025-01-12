import torch.nn as nn
from GDN import GDN
from model_layer import AF_block

# semantic decoder通常使用反卷積或 U-Net 結構還原圖像
# torch.nn.ConvTranspose2d(): 反卷積, 卷積是把A變成B, 反卷積則是把B變成A
def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)

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

# deconv_ResBlock 
class deconv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(deconv_ResBlock, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding)
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out+x
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out 

# deconv_block
class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
        super(deconv_block, self).__init__()
        self.deconv = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding = output_padding)
        self.gdn = nn.GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, activate_func='prelu'): 
        out = self.deconv(x)
        out = self.gdn(out)
        if activate_func=='prelu':
            out = self.prelu(out)
        elif activate_func=='sigmoid':
            out = self.sigmoid(out)
        return out  