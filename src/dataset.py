import os
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import math
import random
import copy
from PIL import Image

import torch
from torch.utils.data import Dataset

from src import utils

class Base(Dataset):
    def __init__(self, params):
        super(Base, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

    def __getitem__(self, index):
        return self.single_view_sampling(index) if self.mode in ['train', 'eval', 'traineval'] else self.arbitrary_sampling(index)

    def __len__(self):
        return self.LEN

    def setup(self):
        if self.mode == 'train':
            self.LEN = self.len
            self.pad = self.radius * self.scale

        elif self.mode == 'eval' or self.mode== 'traineval':
            if self.N_eval is not None:
                self.vals = [int(i * (self.len - 1) / (self.N_eval - 1)) for i in range(self.N_eval)]
                self.LEN = len(self.vals)

            else:
                # self.vals = list(range(self.len))
                # self.LEN = self.len
                if self.mode == 'eval':
                    # self.vals = list(filter(lambda x: x % self.scale != 0, range(self.len)))
                    self.vals = list(range(0, self.len))
                elif self.mode == 'traineval':
                    self.vals = list(filter(lambda x: x % self.scale == 0, range(self.len)))
                self.LEN = len(self.vals)
            self.pad = self.radius * self.scale

        elif self.mode == 'test':
            self.LEN = self.asteps
            self.pad = self.radius

        else:
            print (f'Not {self.mode} mode!')
            exit()

        self.pad = int(max(self.pad, 1))

    def z_trans(self, z):
        return 2 * np.pi * (z + self.pad) / (self.len + 2 * self.pad - 1) - np.pi
    
    def sampling(self, coords, xy_inds, z_coord, pad):
        xy_coords = coords[xy_inds[:, 0] + pad, xy_inds[:, 1] + pad]
        LR_coords = torch.cat([coords[xy_inds[:, 0] + pad, xy_inds[:, 1] + ind][:, 0:1] for ind in [0, pad * 2]], 1)
        TB_coords = torch.cat([coords[xy_inds[:, 0] + ind, xy_inds[:, 1] + pad][:, 1:2] for ind in [0, pad * 2]], 1)
        coords = torch.cat([xy_coords, torch.full((xy_coords.shape[0], 1), z_coord), LR_coords, TB_coords], 1)
        return coords

    def single_view_sampling(self, index):
        j, i = torch.meshgrid(torch.linspace(-np.pi, np.pi, self.H + 2 * self.pad), torch.linspace(-np.pi, np.pi, self.W + 2 * self.pad))
        coords = torch.stack([i, j], -1)

        if self.mode == 'train':
            if self.only_downsampling_in_z:
                xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W))
            else:
                xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H // self.scale), torch.linspace(0, self.W - 1, self.W // self.scale))
            z_ind = index // self.scale * self.scale

            xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
            if xy_inds.shape[0] > self.bsize:
                xy_inds = xy_inds[np.random.choice(xy_inds.shape[0], size=[self.bsize], replace=False)]
                
        elif self.mode == 'traineval':
            if self.only_downsampling_in_z:
                xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W))
            else:
                xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H // self.scale), torch.linspace(0, self.W - 1, self.W // self.scale))
            z_ind = self.vals[index]

            xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
            
        else:
            xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W))
            xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
            z_ind = self.vals[index]

        head, z_coord, tail = [self.z_trans(z) for z in [z_ind - self.pad * self.z_scaler, z_ind, z_ind + self.pad * self.z_scaler]]
        coords = self.sampling(coords, xy_inds, z_coord, self.pad)
        
        data = self.data[z_ind, xy_inds[:, 0], xy_inds[:, 1]]
        # return (data, coords, (np.float32(head), np.float32(tail))) if self.mode == 'train' else (coords, (np.float32(head), np.float32(tail)))
        return (data, coords, torch.tensor([head, tail])) if self.mode == 'train' else (coords, torch.tensor([head, tail]))


    def multi_view_and_scale_sampling(self, zpos, angle, scale):
        H, W, P = [int(self.cam_scale * scale * a) for a in [self.H, self.W, self.pad]]
        j, i = torch.meshgrid(torch.linspace(-self.cam_scale * np.pi, self.cam_scale * np.pi, H + P), torch.linspace(-self.cam_scale * np.pi, self.cam_scale * np.pi, W + P))
        coords = torch.stack([i, j], -1)
        # considering the multi-scale
        t, l = max(0, int((H - self.cam_scale * self.H) / self.cam_scale)), max(0, int((W - self.cam_scale * self.W) / self.cam_scale))
        b, r = max(self.cam_scale * self.H, H) - 1 - t, max(self.cam_scale * self.W, W) - 1 - l
        xy_inds = torch.meshgrid(torch.linspace(t, b, int(self.cam_scale * self.H)), torch.linspace(l, r, int(self.cam_scale * self.W)))
        xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
        th = self.cam_scale * np.pi / (self.len + self.cam_scale * self.pad - 1)
        head, tail = zpos - th, zpos + th
        coords = self.sampling(coords, xy_inds, zpos, int(P / self.cam_scale))
        R = utils.get_rotate_matrix(self.axis, angle)
        _R = torch.linalg.inv(R)
        return coords, (np.float32(head), np.float32(tail)), _R, zpos, angle, scale

    def arbitrary_sampling(self, index):
        return self.multi_view_and_scale_sampling(self.zs[index], self.angles[index], self.scales[index])

class Medical3D(Base):
    def __init__(self, **params):
        super(Medical3D, self).__init__(params)
        self.load_data()
        if self.mode == 'test':
            self.angles = np.linspace(self.angles[0], self.angles[1], self.asteps) if len(self.angles) == 2 else [self.angles[0]] * self.asteps
            self.scales = np.linspace(self.scales[0], self.scales[1], self.asteps) if len(self.scales) == 2 else [self.scales[0]] * self.asteps
            self.zs = np.linspace(self.zpos[0], self.zpos[1], self.asteps) if len(self.zpos) == 2 else [self.zpos[0]] * self.asteps

    def load_data(self, normalise_to_512=True):
        data = self.load_file()
        data = self.nomalize(data)
        self.data = self.align(data)
        self.len, self.H, self.W = self.data.shape
        print (self.len, self.H, self.W)
        
        ## resize the data to 512 x 512 x 512
        if normalise_to_512:
            self.data = self.super_sampling_in_z(self.data)
            self.len, self.H, self.W = self.data.shape
            print (self.len, self.H, self.W)

        ## current orientation should be (z, y, x)
        if self.direction == 'sagittal': ## (x, y, z)
            self.data = self.data.permute(2, 1, 0)
            print(f"direction = {self.direction}, data shape = {self.data.shape}")
            self.len, self.H, self.W = self.data.shape
            print (self.len, self.H, self.W)
        elif self.direction == 'coronal': ## (y, x, z)
            self.data = self.data.permute(1, 2, 0)
            print(f"direction = {self.direction}, data shape = {self.data.shape}")
            self.len, self.H, self.W = self.data.shape
            print (self.len, self.H, self.W) 
        
        self.setup()

    def align(self, data):
        if data.shape[1] != data.shape[2]:
            if data.shape[0] == data.shape[2]:
                data = data.permute(1, 0, 2)

            else:
                data = data.permute(2, 1, 0)
        
        return data

    def load_file(self):
        data = sitk.GetArrayFromImage(sitk.ReadImage(self.file)).astype(float)
        data = torch.from_numpy(data).float().cuda()
        if len(data.shape) == 4:
            modalities = {
                'FLAIR' : 0,
                'T1w'   : 1,
                't1gd'  : 2,
                'T2w'   : 3
            }
            data = data[modalities[self.modality]]
        return data

    def nomalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def getLabel(self):
        if self.mode == "traineval":
            xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H // self.scale), torch.linspace(0, self.W - 1, self.W // self.scale))
            xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
            vals_expanded = torch.tensor(self.vals)[:, None]
            data = self.data[vals_expanded, xy_inds[:, 0], xy_inds[:, 1]]
            return data.reshape(len(self.vals), self.H // self.scale, self.W // self.scale).cpu().numpy()
        return self.data[self.vals].cpu().numpy()

    ## super sampling in z direction to get 512 x 512 x 512
    def super_sampling_in_z(self, data):
        ## generate the nearest z frame
        new_data = torch.linspace(0, 511, 512).float()
        new_data = new_data / 512 * (data.shape[0])
        new_data = new_data.round() ## nearest frame
        
        ## prevent out of bound
        new_data = torch.clamp(new_data, 0, data.shape[0] - 1).long()
        print (new_data)
        
        ## view as 3D for gathering 
        new_data = new_data[..., None, None]
        new_data = new_data.expand(512, 512, 512)
        new_data = torch.gather(data, 0, new_data)
        return new_data