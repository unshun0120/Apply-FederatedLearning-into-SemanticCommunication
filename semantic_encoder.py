import torch.nn as nn
from GDN import GDN
from model_layer import AF_block

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

# The Encoder model with attention feature blocks
class Encoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(Encoder, self).__init__()    
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv//2
        padding_L = (kernel_sz-1)//2
        self.conv1 = conv_ResBlock(3, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv2 = conv_ResBlock(Nc_conv, Nc_conv, use_conv1x1=True, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv3 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.conv4 = conv_ResBlock(Nc_conv, Nc_conv, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.conv5 = conv_ResBlock(Nc_conv, enc_N, use_conv1x1=True, kernel_size = kernel_sz, stride = 1, padding=padding_L)
        self.AF1 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF2 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF3 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF4 = AF_block(Nc_conv, Nh_AF, Nc_conv)
        self.AF5 = AF_block(enc_N, enc_N//2, enc_N)
        self.flatten = nn.Flatten() 
    def forward(self, x, snr):    
        out = self.conv1(x) 
        out = self.AF1(out, snr)
        out = self.conv2(out)   
        out = self.AF2(out, snr)
        out = self.conv3(out) 
        out = self.AF3(out, snr)
        out = self.conv4(out) 
        out = self.AF4(out, snr)
        out = self.conv5(out) 
        out = self.AF5(out, snr)
        out = self.flatten(out)
        return out

# conv_ResBlock  
class conv_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(conv_ResBlock, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride = 1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    def forward(self, x): 
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out+x
        out = self.prelu(out)
        return out 
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gdn = nn.GDN(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x): 
        out = self.conv(x)
        out = self.gdn(out)
        out = self.prelu(out)
        return out
    

