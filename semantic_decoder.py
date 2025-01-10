import torch.nn as nn
from GDN import GDN
from model_layer import AF_block

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,bias=False)

class Decoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_deconv):
        super(Decoder, self).__init__()
        self.enc_shape = enc_shape
        Nh_AF1 = enc_shape[0]//2
        Nh_AF = Nc_deconv//2
        padding_L = (kernel_sz-1)//2
        self.deconv1 = deconv_ResBlock(self.enc_shape[0], Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
        self.deconv2 = deconv_ResBlock(Nc_deconv, Nc_deconv, use_deconv1x1=True, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding = 1)
        self.deconv3 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv4 = deconv_ResBlock(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv5 = deconv_ResBlock(Nc_deconv, 3, use_deconv1x1=True, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.AF1 = AF_block(self.enc_shape[0], Nh_AF1, self.enc_shape[0])
        self.AF2 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF3 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF4 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
        self.AF5 = AF_block(Nc_deconv, Nh_AF, Nc_deconv)
    def forward(self, x, snr):  
        print(self.enc_shape)
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = self.AF1(out, snr)
        out = self.deconv1(out) 
        out = self.AF2(out, snr)
        out = self.deconv2(out) 
        out = self.AF3(out, snr)
        out = self.deconv3(out)
        out = self.AF4(out, snr)
        out = self.deconv4(out)
        out = self.AF5(out, snr)
        out = self.deconv5(out, 'sigmoid')  
        return out

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