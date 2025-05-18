import torch
from torch import nn
import torch.nn.functional as F
from . import base
import math

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

# class NeRFMLP(base.baseModel):
#     def __init__(self, **params):
#         super(NeRFMLP, self).__init__(params)
#         self.coords_MLP = nn.ModuleList(
#             [nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD - 1)]]
#         )
#         for idx, mlp in enumerate(self.coords_MLP):
#             if idx in self.skips:
#                 mlp.requires_grad_(False)
#         self.out_MLP = nn.Linear(self.netW, self.out_ch)

#     def forward(self, x):
#         x = self.embed(x)
#         h = x
#         for idx, mlp in enumerate(self.coords_MLP):
#             h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h)) 
#         out = self.out_MLP(h)
#         return out 

class NeRFMLP(base.baseModel):
    def __init__(self, **params):
        super(NeRFMLP, self).__init__(params)
        ## 9 layers. 
        ## skipping 4th, 9th 
        ## during construction, skip index the width is increased 
        ## during forward pass, skip index is concatenation 
        ## 
        ## for below, same level is the same layer but different index during construction and forward pass
        ## 
        ## during contruction, during foward pass, 
        ## 63 -> 256            0th 
        ## 0th                  1th 
        ## 1th                  2nd 
        ## 2nd                  3rd 
        ## 3rd                  4th concatenation, skipped
        ## 4th wider input      5th 
        ## 5th                  6th
        ## 6th                  7th
        ## 7th                  8th 
        ## 8th                  9th concatenation, skipped
        ## 9th wider input      10th
        ## 256 -> 128
        ## 128 -> 2 
        
        self.coords_MLP = nn.ModuleList(
            [nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD-1+len(self.skips))]]
        )
        ## skip the gradients of the skipped layers
        for idx, mlp in enumerate(self.coords_MLP):
            if idx in self.skips:
                mlp.requires_grad_(False)
        
        ## 612738 parameters        
        # self.preout_MLP = nn.Linear(self.netW, self.netW // 2)
        # self.out_MLP = nn.Linear(self.netW // 2, self.out_ch)

        ## 580098 parameters
        self.out_MLP = nn.Linear(self.netW, self.out_ch)

    def forward(self, x):
        x = self.embed(x)
        h = x
        for idx, mlp in enumerate(self.coords_MLP):
            h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h)) 
            
        # h = F.relu(self.preout_MLP(h))
        out = self.out_MLP(h)
        return out 

## just replace the mlp within RA-CT NeRF 
class NGPMLP(torch.nn.Module):
    def __init__(self, **params):
        super(NGPMLP, self).__init__()
        for k, v in params.items():
            setattr(self, k, v)
        # self.model = tcnn.NetworkWithInputEncoding(n_input_dims=self.n_input_dims, 
        #                                            n_output_dims=self.n_output_dims, 
        #                                            encoding_config=self.encoding, 
        #                                            network_config=self.network,
        #                                            )

        self.encoding = tcnn.Encoding(n_input_dims=self.n_input_dims, encoding_config=self.encoding)
        self.network = tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=self.n_output_dims, network_config=self.network)
        self.model = torch.nn.Sequential(self.encoding, self.network)
 
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.n_input_dims)
        y = self.model(x)
        y = y.reshape(*shape[:-1], self.n_output_dims)
        return y
    
class NGPModel(torch.nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn):
        super(NGPModel, self).__init__()
        print("Using NGPModel")
        self.coarse = coarse
        self.sample_fn = sample_fn
        self.render_fn = render_fn
        self.imp_fn = imp_fn

    def forward(self, x):
        coords, depths = x
        # coords = coords.squeeze(0)
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5
        
        return self.Render(coords, depths, is_train=True)

    def eval_forward(self, x):
        coords, depths = x        
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5
        return self.Render(coords, depths, is_train=False)
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        # ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        # raw = self.fine(ans['pts'])
        # out = self.render_fn(raw, **ans)

        # out0['rgb'] = out0['rgb'].clamp(0.0, 1.0)
        return out0['rgb'], list(self.coarse.network.parameters())[0]
    
    

class FullModel(torch.nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn):
        super(FullModel, self).__init__()
        print("Using FullModel")
        self.coarse = coarse
        self.fine = fine
        self.sample_fn = sample_fn
        self.render_fn = render_fn
        self.imp_fn = imp_fn

    def forward(self, x):
        coords, depths = x
        # coords = coords.squeeze(0)
        
        return self.Render(coords, depths, is_train=True)

    def eval_forward(self, x):
        coords, depths = x        
        return self.Render(coords, depths, is_train=False)
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        raw = self.fine(ans['pts'])
        out = self.render_fn(raw, **ans)
        return out['rgb'], out0['rgb']


class VcubeMLP(torch.nn.Module):
    def __init__(self, **params):
        super(VcubeMLP, self).__init__()
        for k, v in params.items():
            setattr(self, k, v)
        # self.model = tcnn.NetworkWithInputEncoding(n_input_dims=self.n_input_dims, 
        #                                            n_output_dims=self.n_output_dims, 
        #                                            encoding_config=self.encoding, 
        #                                            network_config=self.network,
        #                                            )

        self.encoding = tcnn.Encoding(n_input_dims=self.n_input_dims, encoding_config=self.encoding)
        self.colourNetwork = tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=self.n_output_dims, network_config=self.colour_network)
        self.sizeNetwork = tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=3, network_config=self.size_network)
        self.colourModel = torch.nn.Sequential(self.encoding, self.colourNetwork)
        self.sizeModel = torch.nn.Sequential(self.encoding, self.sizeNetwork)

        self.freeze_encoding()
        self.freeze_colourNetwork()
 
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.n_input_dims)
        y = self.colourModel(x)
        y = y.reshape(*shape[:-1], self.n_output_dims)
        return y

    def forward_size(self, x):
        shape = x.shape
        x = x.reshape(-1, self.n_input_dims)
        y = self.sizeModel(x)
        # y = F.sigmoid(y)
        # y = F.relu(y) + 0.0001
        y = y.reshape(*shape[:-1], 3)
        return y
    
    def network_parameters(self):
        # return list(self.colourNetwork.parameters())
        return list(self.colourNetwork.parameters()) + list(self.sizeNetwork.parameters())

    # colourNetwork
    def freeze_colourNetwork(self):
        for param in self.colourNetwork.parameters():
            param.requires_grad = False

    def unfreeze_colourNetwork(self):
        for param in self.colourNetwork.parameters():
            param.requires_grad = True

    # sizeNetwork
    def freeze_sizeNetwork(self):
        for param in self.sizeNetwork.parameters():
            param.requires_grad = False

    def unfreeze_sizeNetwork(self):
        for param in self.sizeNetwork.parameters():
            param.requires_grad = True

    # encoding network
    def freeze_encoding(self):
        for param in self.encoding.parameters():
            param.requires_grad = False

    def unfreeze_encoding(self):
        for param in self.encoding.parameters():
            param.requires_grad = True

    def freeze_all(self):
        self.freeze_colourNetwork()
        self.freeze_sizeNetwork()
        self.freeze_encoding()

    def unfreeze_all(self):
        self.unfreeze_colourNetwork()
        self.unfreeze_sizeNetwork()
        self.unfreeze_encoding()

class VcubeModel(torch.nn.Module):
    def __init__(self, coarse, fine, sample_fn, render_fn, imp_fn, alternating_training=True):
        super(VcubeModel, self).__init__()
        print("Using VcubeModel")
        self.coarse = coarse
        self.render_fn = render_fn
        self.n_samples = coarse.n_samples

        methods = {"size": self.sampling_with_size, "scale": self.sampling_with_scale}
        self.sampling = methods[coarse.size_network_method]

        ## for alternating training
        self.first_cycle = False
        self.training_modes = coarse.training_modes
        self.training_mode_max_iters = coarse.training_mode_max_iters
        self.training_mode_idx = 0
        self.cnt = 0

        if alternating_training:
            print("Using alternating training")
            self.first_cycle = True

    def forward(self, x):
        coords, depths = x
        # coords = coords.squeeze(0)
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5
        
        return self.Render(coords, depths, is_train=True)

    def eval_forward(self, x):
        coords, depths = x        
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5
        return self.Render(coords, depths, is_train=False)
    
    def eval_size(self, x):
        coords, depths = x
        coords = coords / (2*math.pi) + 0.5
        depths = depths / (2*math.pi) + 0.5

        ## dic = self.sampling_with_size(coords, depths, is_train=False)
        batch = coords
        B = batch.shape[0]
        # n_cnts = batch.shape[1]
        (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), torch.split(depths, [1, 1], dim=-1)

        # steps = round(math.pow(self.n_samples, 1./3) + 1)
        # t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
        # t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3) ## (64, 3)

        dxdydz = self.coarse.forward_size(cnts)
        dxdydz = F.sigmoid(dxdydz)
        # print(dxdydz)
        # dx, dy, dz = torch.split(dxdydz, [1,1,1], dim=-1)
        
        # return torch.tensor((dic['dx'], dic['dy'], dic['dz']))
        return dxdydz
    
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sampling(coord_batch, depths, is_train=is_train, R=R)
        raw0 = self.coarse(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        
        # ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        # raw = self.fine(ans['pts'])
        # out = self.render_fn(raw, **ans)

        # out0['rgb'] = out0['rgb'].clamp(0.0, 1.0)
        return out0['rgb'], list(self.coarse.network_parameters())[0]
    
    def sampling_with_size(self, batch, depths, is_train=False, R=None):
        ## batch comes in size [B, b, 7]
        B = batch.shape[0]
        n_cnts = batch.shape[1]
        (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), torch.split(depths, [1, 1], dim=-1)

        steps = round(math.pow(self.n_samples, 1./3) + 1)
        t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
        t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3) ## (64, 3)

        if self.first_cycle:
            left, right = torch.split(LR, [1, 1], dim=-1) ## (B, b, 1)
            top, bottom = torch.split(TB, [1, 1], dim=-1)
            l, r = left.expand([B, n_cnts, self.n_samples]), right.expand([B, n_cnts, self.n_samples])
            t, b = top.expand([B, n_cnts, self.n_samples]), bottom.expand([B, n_cnts, self.n_samples])
            n = near[:, None].expand([B, n_cnts, self.n_samples])
            f = far[:, None].expand([B, n_cnts, self.n_samples])
            dx, dy, dz = (r - l)/2, (t - b)/2, (f - n)/2
        else:
            dxdydz = self.coarse.forward_size(cnts)
            dxdydz = F.sigmoid(dxdydz)
            # print(dxdydz)
            dx, dy, dz = torch.split(dxdydz, [1,1,1], dim=-1)
        cx, cy, cz = torch.split(cnts, [1,1,1], dim=-1)

        ## to get the range for sampling 
        x_l, x_r = cx - dx, cx + dx
        y_l, y_r = cy - dy, cy + dy
        z_l, z_r = cz - dz, cz + dz

        
        # left, right = torch.split(LR, [1, 1], dim=-1) ## (B, b, 1)
        # top, bottom = torch.split(TB, [1, 1], dim=-1)
        # steps = round(math.pow(self.n_samples, 1./3) + 1)
        # t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
        # t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3) ## (64, 3)

        # x_l, x_r = left.expand([B, n_cnts, self.n_samples]), right.expand([B, n_cnts, self.n_samples])
        # y_l, y_r = top.expand([B, n_cnts, self.n_samples]), bottom.expand([B, n_cnts, self.n_samples])
        # # z_l = torch.full_like(x_l, 1.).view(-1, n_samples) * near[:, None]
        # # z_r = torch.full_like(x_r, 1.).view(-1, n_samples) * far[:, None]
        # z_l = near[:, None].expand([B, n_cnts, self.n_samples])
        # z_r = far[:, None].expand([B, n_cnts, self.n_samples])

        if is_train:
            x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(B, n_cnts, self.n_samples) - t_vals[:, 0] / 2 * (x_r - x_l)
            y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(B, n_cnts, self.n_samples) - t_vals[:, 0] / 2 * (y_r - y_l)
            z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(B, n_cnts, self.n_samples) - t_vals[:, 0] / 2 * (z_r - z_l)
                
        else:
            # print("Using is_train=False")
            x_vals = x_l + t_vals[:, 0] * (x_r - x_l) - t_vals[:, 0] / 2 * (x_r - x_l)
            y_vals = y_l + t_vals[:, 1] * (y_r - y_l) - t_vals[:, 0] / 2 * (y_r - y_l)
            z_vals = z_l + t_vals[:, 2] * (z_r - z_l) - t_vals[:, 0] / 2 * (z_r - z_l)

        pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
        if R is not None:
            pts, cnts = pts @ R, cnts @ R

        ## TODO
        ## change dx, dy, dz dim
        return {'pts' : pts, 'cnts' : cnts, 'dx' : dx, 'dy' : dy, 'dz' : dz}


    def sampling_with_scale(self, batch, depths, is_train=False, R=None):
        ## batch comes in size [B, b, 7]
        B = batch.shape[0]
        n_cnts = batch.shape[1]
        (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), torch.split(depths, [1, 1], dim=-1)

        steps = round(math.pow(self.n_samples, 1./3) + 1)
        t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
        t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3) ## (64, 3)

        ## get the left, right, top, bottom
        left, right = torch.split(LR, [1, 1], dim=-1) ## (B, b, 1)
        top, bottom = torch.split(TB, [1, 1], dim=-1)
        l, r = left.expand([B, n_cnts, self.n_samples]), right.expand([B, n_cnts, self.n_samples])
        t, b = top.expand([B, n_cnts, self.n_samples]), bottom.expand([B, n_cnts, self.n_samples])
        n = near[:, None].expand([B, n_cnts, self.n_samples])
        f = far[:, None].expand([B, n_cnts, self.n_samples])


        if self.first_cycle:
            sxsysz = torch.ones((3,))
        else:
            sxsysz = self.coarse.forward_size(cnts)
            sxsysz = F.relu(sxsysz) + 0.001 ## activation function

        # print(sxsysz)
        sx, sy, sz = torch.split(sxsysz, [1,1,1], dim=-1)
        cx, cy, cz = torch.split(cnts, [1,1,1], dim=-1)

        ## to get the range for sampling 
        x_l, x_r = cx - sx * (cx - l), cx + sx * (r - cx)
        y_l, y_r = cy - sy * (cy - b), cy + sy * (t - cy)
        z_l, z_r = cz - sz * (cz - n), cz + sz * (f - cz)

        dx = x_r - x_l
        dy = y_r - y_l
        dz = z_r - z_l

        if is_train:
            x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(B, n_cnts, self.n_samples) - t_vals[:, 0] / 2 * (x_r - x_l)
            y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(B, n_cnts, self.n_samples) - t_vals[:, 0] / 2 * (y_r - y_l)
            z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(B, n_cnts, self.n_samples) - t_vals[:, 0] / 2 * (z_r - z_l)
                
        else:
            # print("Using is_train=False")
            x_vals = x_l + t_vals[:, 0] * (x_r - x_l) - t_vals[:, 0] / 2 * (x_r - x_l)
            y_vals = y_l + t_vals[:, 1] * (y_r - y_l) - t_vals[:, 0] / 2 * (y_r - y_l)
            z_vals = z_l + t_vals[:, 2] * (z_r - z_l) - t_vals[:, 0] / 2 * (z_r - z_l)

        pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
        if R is not None:
            pts, cnts = pts @ R, cnts @ R

        ## TODO
        ## change dx, dy, dz dim
        return {'pts' : pts, 'cnts' : cnts, 'dx' : dx, 'dy' : dy, 'dz' : dz}
    
    def alternating_training(self, idx):
        training_mode = self.training_modes[self.training_mode_idx]

        if self.cnt == 0:
            if training_mode == "colour":
                print("Training colour and encoding")
                self.coarse.unfreeze_all()
                self.coarse.freeze_sizeNetwork()
            elif training_mode == "size":
                print("Training size")
                self.coarse.unfreeze_all()
                self.coarse.freeze_colourNetwork()
                self.coarse.freeze_encoding()

        self.cnt += 1

        cur_max_iter = self.training_mode_max_iters[self.training_mode_idx]
        if self.cnt == cur_max_iter:
            self.cnt = 0
            self.training_mode_idx = (self.training_mode_idx + 1) % len(self.training_modes)
            if self.first_cycle:
                self.first_cycle = False
    
