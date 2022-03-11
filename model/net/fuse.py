
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pyramid-Context Guided Fusion Module
class PCGF_Module(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 2, 4, 8), descriptor=8, mid_feats=16, sp_feats='u'):
        super().__init__()
        print('[PCGF]: sp = %s, pp = %s, t = %d, m = %d.' % (sp_feats, pp_size, descriptor, mid_feats))
        
        self.sp_feats = sp_feats
        self.pp_size = pp_size
        self.feats_size = sum([(s ** 2) for s in self.pp_size])
        self.descriptor = descriptor

        # without dim reduction
        if (descriptor == -1) or (self.feats_size < descriptor):
            self.des = nn.Identity()
            self.fc = nn.Sequential(nn.Linear(in_feats * self.feats_size, mid_feats, bias=False),
                                    nn.BatchNorm1d(mid_feats),
                                    nn.ReLU(inplace=True))
        # with dim reduction
        else:
            self.des = nn.Conv2d(self.feats_size, self.descriptor, kernel_size=1)
            self.fc = nn.Sequential(nn.Linear(in_feats * descriptor, mid_feats, bias=False),
                                    nn.BatchNorm1d(mid_feats),
                                    nn.ReLU(inplace=True))

        self.fc_x = nn.Linear(mid_feats, in_feats)
        self.fc_y = nn.Linear(mid_feats, in_feats)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size, ch, _, _ = x.size()
        sp_dict = {'x': x, 'y': y, 'u': x+y}

        pooling_pyramid = []
        for s in self.pp_size:
            l = F.adaptive_avg_pool2d(sp_dict[self.sp_feats], s).view(batch_size, ch, 1, -1)
            pooling_pyramid.append(l)                       # [b, c, 1, s^2]
        z = torch.cat(tuple(pooling_pyramid), dim=-1)       # [b, c, 1, f]
        z = z.reshape(batch_size * ch, -1, 1, 1)            # [bc, f, 1, 1]
        z = self.des(z).view(batch_size, -1)                # [bc, d, 1, 1] => [b, cd]
        z = self.fc(z)                                      # [b, m]

        z_x, z_y = self.fc_x(z), self.fc_y(z)               # [b, c]      
        w_x, w_y = self.sigmoid(z_x), self.sigmoid(z_y)     # [b, c]
        rf_x = x * w_x.view(batch_size, ch, 1, 1)           # [b, c, h, w]
        rf_y = y * w_y.view(batch_size, ch, 1, 1)           # [b, c, h, w]
        out_feats = rf_x + rf_y                             # [b, c, h, w]

        return out_feats, rf_x, rf_y

# Multi-Level General Fusion Module
class MLGF_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='par', att_setting={}):
        super().__init__()
        module_dict = {
            'se': SE_Block,
            'par': PAR_Block,
            'idt': IDT_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Fuse_Block(in_feats, **fuse_setting)
    
    def forward(self, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(x, y), x, y

class General_Fuse_Block(nn.Module):
    def __init__(self, in_feats, merge_mode='grp', init=True, civ=1):
        super().__init__()
        merge_dict = {
            'add': Add_Merge(in_feats),
            'cc3': CC3_Merge(in_feats),
            'lma': LMA_Merge(in_feats),
            'grp': nn.Conv2d(2*in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats)
        }
        self.merge_mode = merge_mode
        self.merge = merge_dict[merge_mode]
        if init and isinstance(self.merge, nn.Conv2d):
            self.merge.weight.data.fill_(civ)
        
    def forward(self, x, y):
        if self.merge_mode != 'grp':
            return self.merge(x, y)
        b, c, h, w = x.size()
        feats = torch.cat((x, y), dim=-2).reshape(b, 2*c, h, w)   # [b, c, 2h, w] => [b, 2c, h, w]
        return self.merge(feats)

# Attention Refinement Blocks

class SE_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // r, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x
    
class PAR_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, mid_feats=16):
        super().__init__()
        self.layer_size = pp_layer                  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor                # d: descriptor num (for one channel)

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)   # [b,  c, 1, f]
        y = y.reshape(b*c, f, 1, 1)                     # [bc, f, 1, 1]
        y = self.des(y).view(b, c*d)                    # [bc, d, 1, 1] => [b, cd, 1, 1]
        w = self.mlp(y).view(b, c, 1, 1)                # [b,  c, 1, 1] => [b, c, 1, 1]
        return w * x

class IDT_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

# Merge Modules

class Add_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x, y):
        return x+y

class LMA_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        return x + self.lamb * y

class CC3_Merge(nn.Module):
    def __init__(self, in_feats, *args, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2*in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))

class ADD_Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x+y, x, y

# Constant that stores available fusion modules
FUSE_MODULE_DICT = {
    'add': ADD_Module,
    'mlgf': MLGF_Module,
    'pcgf': PCGF_Module
}