import SimpleITK as sitk
import torch    
import argparse
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch.nn as nn 
import torch.nn.functional as F

import time
from PIL import Image

def load_file(filepath):
    data = sitk.GetArrayFromImage(sitk.ReadImage(filepath)).astype(float)
    data = torch.from_numpy(data).float().cuda()
    return data

def super_sampling_in_z(data, z_size=512):
    ## generate the nearest z frame
    new_data = torch.linspace(0, z_size-1, z_size).float()
    new_data = new_data / z_size * (data.shape[0])
    new_data = new_data.round() ## nearest frame
    
    ## prevent out of bound
    new_data = torch.clamp(new_data, 0, data.shape[0] - 1).long()
    print (new_data)
    
    ## view as 3D for gathering 
    new_data = new_data[..., None, None]
    new_data = new_data.expand(z_size, 512, 512)
    new_data = torch.gather(data, 0, new_data)
    return new_data

def load_data(filepath, supersample=False):
    start_time = time.time()
    data = load_file(filepath)
    data = nomalize(data)
    data = align(data)
    length, H, W = data.shape
    print(length, H, W)
    
    if supersample:
        data = super_sampling_in_z(data)
        length, H, W = data.shape
        print (length, H, W)
    
    end_time = time.time()
    print(f"Time taken for loading data is {end_time - start_time} seconds")
    return data

def align(data):
    if data.shape[1] != data.shape[2]:
        if data.shape[0] == data.shape[2]:
            data = data.permute(1, 0, 2)
        else:
            data = data.permute(2, 1, 0)
    return data

def nomalize(data):
    return (data - data.min()) / (data.max() - data.min())

def calculate_metrics(current_frame, previous_frame):
    psnr_value = peak_signal_noise_ratio(previous_frame, current_frame, data_range=1)
    # ssim_value = structural_similarity(previous_frame, current_frame, data_range=current_frame.max() - current_frame.min())
    ssim_value = structural_similarity(current_frame, previous_frame, win_size=11, data_range=1, channel_axis=0)
    return psnr_value, ssim_value

def visualize(data, n_eval, scale):
    data = data.cpu()
    length, _, _ = data.shape
    eval_indices = [int(i * (length - 1) / (n_eval - 1)) for i in range(n_eval)]
    train_indices = list(filter(lambda x: x % scale == 0, range(length)))

    previous_frame = None

    for i in range(length):
        # Convert tensor to numpy and scale to [0, 255] for visualization
        frame = data[i].cpu().numpy() * 255
        frame = frame.astype(np.uint8)
        
        # Convert grayscale to BGR for display with colored text
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Determine if the frame is for evaluation or training
        label0 = f"Evaluation @ 50" if i in eval_indices else ""
        label1 = f"Training" if i not in train_indices else ""
        
        # Add the label to the frame
        cv2.putText(frame_bgr, label0, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, label1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Calculate PSNR and SSIM if there is a previous frame
        if previous_frame is not None:
            psnr_value, ssim_value = calculate_metrics(frame, previous_frame)
            psnr_text = f"PSNR: {psnr_value:.2f}"
            ssim_text = f"SSIM: {ssim_value:.2f}"
            cv2.putText(frame_bgr, psnr_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, ssim_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Frame', frame_bgr)

        # Wait for a key press to move to the next frame
        key = cv2.waitKey(0)
        if key == ord('q'):  # Press 'q' to quit early
            break

        # Update the previous frame
        previous_frame = frame

    cv2.destroyAllWindows()
    

def downsample(data, scale):
    # return data[::scale, ::scale, ::scale]

    H, W = data.shape[1], data.shape[2]
    xy_inds = torch.meshgrid(torch.linspace(0, H - 1, H // scale), torch.linspace(0, W - 1, W // scale))
    xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
    
    a = torch.linspace(0, H - 1, H // scale).long()
    b = torch.linspace(0, H - scale, H // scale).long()
    print(a)
    print(b)
    print(a == b)
    
    vals = list(filter(lambda x: x %scale == 0, range(data.shape[0])))
    vals_expanded = torch.tensor(vals)[:, None]
    data = data[vals_expanded, xy_inds[:, 0], xy_inds[:, 1]]
    data = data.reshape(len(vals), H // scale, W // scale)
    return data


def bicubic_gt(data, scale):
    data = data.cpu().numpy()
    B, H, W = data.shape
    
    new_data = np.zeros((B*2, 2*H, 2*W))
    
    for i, img in enumerate(data):
        img = Image.fromarray(img)
        img = img.resize((2*H, 2*W), Image.BICUBIC)
        new_data[i*2] = np.array(img)
        
    return new_data
    
## TODO only works for scale in power of 2
def bicubic(data, scale):
    start_time = time.time()
    ## compute 1D bicubic interpolation. 
    t = 0.5
    A = torch.tensor([t**i for i in range(4)])
    B = torch.tensor([[0, 2, 0, 0], [-1, 0, 1 , 0], [2, -5, 4, -1], [-1, 3, -3, 1]]).float()
    T = 1/2 * A @ B ## (1, 4) for  convolution. 
    T = T[None, None, ...]
    
    conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=0)
    with torch.no_grad():
        conv.weight = nn.Parameter(T)
        conv.bias = nn.Parameter(torch.zeros(1))
    
        ## data is B, H, W
        ts = scale
        scale = 2
        while ts > 1:
            ts = ts // 2
            
            B, H, W = data.shape
            newB, newH, newW = B*2, H*2, W*2
            
            ## do 1D bicubic for x first
            new_data = torch.zeros(B, H, newW)
            new_data[::, ::, ::scale] = data
            
            data = data.reshape(B*H, 1, W)
            data = F.pad(data, (1, 2), mode='constant', value=0)
            data = conv(data)
            data = data.reshape(B, H, W)
            new_data[::, ::, 1::scale] = data
            
            ## do 1D bicubic for y
            data = new_data ## B, H, 2*W
            new_data = torch.zeros(B, newH, newW)
            new_data[::, ::scale, ::] = data
            
            data = data.permute(0, 2, 1) ## B, 2W, H
            data = data.reshape(B*newW, 1, H)
            data = F.pad(data, (1, 2), mode='constant', value=0)
            data = conv(data)
            data = data.reshape(B, newW, H)
            data = data.permute(0, 2, 1)
            
            new_data[::, 1::scale, ::] = data
            
            ## do it in the z direction
            data = new_data 
            new_data = torch.zeros(newB, newH, newW)
            new_data[::scale, ::, ::] = data
            
            ## B, H, W => W, H, B
            data = data.permute(2, 1, 0)
            data = data.reshape(newW*newH, 1, B)
            data = F.pad(data, (1, 2), mode='constant', value=0)
            data = conv(data)
            data = data.reshape(newW, newH, B)
            data = data.permute(2, 1, 0)
            
            new_data[1::scale, ::, ::] = data

            data = new_data
        
        end_time = time.time()
        print(f"Time taken for bicubic interpolation is {end_time - start_time} seconds")

    new_data = torch.clamp(new_data, 0., 1.)
    return new_data
        
def save(gt, pd, save_path, scale):
    if not os.path.exists(os.path.join(save_path, "eval")):
        os.makedirs(os.path.join(save_path, "eval"))
    n = gt.shape[0]
    psnrs = []
    ssims = []

    psnrs_training = []
    ssims_training = []
    
    for i in range(n):
        img_gt = gt[i]
        img = pd[i]
        
        img_gt_01 = img_gt
        img_01 = img
        
        img = np.uint8(img*255.)
        img_gt = np.uint8(img_gt*255.)
        
        cv2.imwrite(os.path.join(save_path, "eval", f'{str(i).zfill(len(str(n)))}_gt.png'), img_gt)
        cv2.imwrite(os.path.join(save_path, "eval", f'{str(i).zfill(len(str(n)))}_ours.png'), img)
        
        psnr_value, ssim_value = calculate_metrics(img_01, img_gt_01)
        psnrs.append(psnr_value)
        ssims.append(ssim_value)
        
        if i % scale == 0:
            psnrs_training.append(psnr_value)
            ssims_training.append(ssim_value)
    
    with open(os.path.join(save_path, 'logs.txt'), 'w') as f:
        f.write(f"PSNR: {np.mean(psnrs)}\n")
        f.write(f"SSIM: {np.mean(ssims)}\n")
        f.write(f"PSNR Training: {np.mean(psnrs_training)}\n")
        f.write(f"SSIM Training: {np.mean(ssims_training)}\n")
        
        print(f"PSNR: {np.mean(psnrs)}")
        print(f"SSIM: {np.mean(ssims)}")
        print(f"PSNR Training: {np.mean(psnrs_training)}")
        print(f"SSIM Training: {np.mean(ssims_training)}")
        
    print(f"Images saved at {save_path}")


def main(args):
    filepath = os.path.join("kits19/data/", args.case,'imaging.nii.gz')
    print("Loading data from", filepath)
    data = load_data(filepath)
    print("Data loaded successfully")
    
    down = downsample(data, args.scale)
    mine = bicubic(down, args.scale)
    # ref = bicubic_gt(down, args.scale)

    # visualize(mine, args.n_eval, args.scale)

    save_path = os.path.join(f"{args.save_folder}/RA-CT NeRFx{args.scale}", args.case)
    save(data.cpu().numpy(), mine.cpu().numpy(), save_path, args.scale)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case_00000")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--save_folder", type=str, default="save_bicubic-tmp")
    args = parser.parse_args()
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main(args)