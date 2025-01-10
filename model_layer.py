import torch.nn
import torch.nn as nn


# AF_block
class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin+1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, snr):
        # out = F.adaptive_avg_pool2d(x, (1,1)) 
        # out = torch.squeeze(out)
        # out = torch.cat((out, snr), 1)
        if snr.shape[0] > 1:
            snr = snr.squeeze()
        snr = snr.unsqueeze(1)  
        mu = torch.mean(x, (2, 3))
        out = torch.cat((mu, snr), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out*x
        return out

# Channel_VLC  
def Channel_VLC(z, snr, cr, channel_type = 'AWGN'):
    z = Power_norm_VLC(z, cr)
    if channel_type == 'AWGN':
        z = AWGN_channel_VLC(z, snr, cr)
    return z

# Power_norm_VLC
def Power_norm_VLC(z, cr, P = 1):
    batch_size, z_dim = z.shape
    Kv = torch.ceil(z_dim*cr).int()
    z_power = torch.sqrt(torch.sum(z**2, 1))
    z_M = z_power.repeat(z_dim, 1).cuda()
    return torch.sqrt(Kv*P)*z/z_M.t()

# AWGN_channel_VLC
def AWGN_channel_VLC(x, snr, cr, P = 1):  
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    mask = mask_gen(length, cr).cuda()
    noise = torch.sqrt(P/gamma)*torch.randn(1, length).cuda()
    noise = noise*mask
    y = x+noise
    return y

# mask_gen
def mask_gen(N, cr, ch_max = 48):
    MASK = torch.zeros(cr.shape[0], N).int()
    nc = N//ch_max
    for i in range(0, cr.shape[0]):
        L_i = nc*torch.round(ch_max*cr[i]).int()
        MASK[i, 0:L_i] = 1
    return MASK   