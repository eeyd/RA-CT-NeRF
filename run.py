import argparse
from tqdm import tqdm
from functools import reduce

import math
import random
import numpy as np
import os 
import torch.multiprocessing as mp
from torch.distributed import init_process_group

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from src import Cfg, utils

import time


def argParse():
    parser = argparse.ArgumentParser(description='MISR3D')
    # basic settings
    parser.add_argument('expname', type=str)
    parser.add_argument('--cfg', default='configs/example.yaml')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--mode', choices=['train', 'eval', 'test', 'eval_size'], default='train')
    parser.add_argument('--file', type=str)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--eval_iter', type=int)
    parser.add_argument('--N_eval', type=int) # number of eval imgs
    parser.add_argument('--save_map', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_type', default='current')

    # test options
    parser.add_argument('--zpos', nargs='+', type=float) # pos of the medical volume
    parser.add_argument('--scales', nargs='+', type=float) # rendering scale
    parser.add_argument('--angles', nargs='+', type=int) # init rendering angle
    parser.add_argument('--axis', nargs='+', type=int) # rotation axis, e.g., [1,1,0]
    parser.add_argument('--cam_scale', nargs='+', type=float) # camera size = img size * cam_scale
    parser.add_argument('--is_details', action='store_true') # save the details of rendering
    parser.add_argument('--is_gif', action='store_true') # save gif
    parser.add_argument('--is_video', action='store_true') # save video
    parser.add_argument('--asteps', type=int)

    # other options
    parser.add_argument('--modality', choices=['FLAIR', 'T1w', 't1gd', 'T2w']) # modality for mris
    parser.add_argument('--workers', type=int) # workers for saving imgs
    parser.add_argument('--multi_gpu', action='store_true') # multi-gpu
    args = parser.parse_args()

    return args


def train(cfg):
    while cfg.i_step <= cfg.max_iter:
        for batch in cfg.trainloader:

            ## freeze network for training.
            if cfg.alternating_training == True:
                if cfg.multi_gpu:
                    cfg.fullmodel.module.alternating_training(cfg.i_step)
                else:
                    cfg.fullmodel.alternating_training(cfg.i_step)
            elif cfg.alternating_training == False:
                cfg.fullmodel.module.coarse.unfreeze_all()

            ## training pipeline.
            cfg.optim.zero_grad()
            gts, coords, depths = batch
            # gts, coords = gts.squeeze(0), coords.squeeze(0)
            # gts = gts.squeeze(0)
            # rgb, rgb0 = cfg.Render(coords, depths, is_train=True)
            rgb, rgb0 = cfg.fullmodel((coords, depths))
            loss = cfg.loss_fn(rgb, rgb0, gts)
            # print(f"[GPU {cfg.rank}] Loss: {loss.item()}")
            loss.backward()
            # for name, param in cfg.model.named_parameters():
            #     if param.requires_grad and param.grad is None:
            #         print(f"Parameter {name} is not used in the forward pass.")
            # for name, param in cfg.model_ft.named_parameters():
            #     if param.requires_grad and  param.grad is None:
            #         print(f"Parameter {name} is not used in the forward pass.")
            # cfg.Update_grad()
            cfg.optim.step()

            # if cfg.i_step % 500 == 0:
            #     for param_group in cfg.optim.param_groups:
            #         print(f"[GPU {cfg.rank}] Learning rate: {param_group['lr']}")

            with torch.no_grad():
                cfg.Update_lr()

            if not cfg.multi_gpu or (cfg.multi_gpu and cfg.rank == 0):
                with torch.no_grad():
                    cfg.Update(loss, rgb.cpu().numpy(), gts.cpu().numpy())
                    if cfg.i_step % cfg.log_iter == 0: cfg.Log()
                    if cfg.i_step % cfg.save_iter == 0: cfg.Save()
               
            ## always execute the following code, but non-major ranks do not save.     
            with torch.no_grad():
                to_save = not cfg.multi_gpu or (cfg.multi_gpu and cfg.rank == 0)
                if cfg.i_step % cfg.eval_iter == 0: globals()['eval'](cfg, to_save=to_save) ## do this one first to ensure when self.psnr for traineval too will be set to true when metric improves.
                if cfg.i_step % cfg.eval_iter == 0: globals()['traineval'](cfg, to_save=to_save)
                # if cfg.i_step % cfg.eval_iter == 0: globals()['eval_size'](cfg, to_save=to_save)
                
                cfg.pbar.update(1)

                # update step and pbar
            if (cfg.i_step > cfg.max_iter) or (cfg.resume and cfg.i_step == cfg.max_iter): 
                print(f"[GPU{cfg.rank}]Exiting...")
                return
        
            cfg.i_step += 1
                
                

def eval(cfg, to_save=True):
    N, W, H, S = cfg.evalset.__len__(), cfg.evalset.W, cfg.evalset.H, cfg.bs_eval
    pds = np.zeros((N, W * H))
    dataloader = tqdm(cfg.evalloader)

    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            dataloader.set_description(f'[EVAL] : {idx}')
            coords, depths = batch
            # coords = coords.squeeze(0)
            # for cidx in range(math.ceil(W * H / S)):
            #     select_coords = coords[list(range(S * cidx, min(S * (cidx + 1), len(coords))))]
            #     rgb, _ = cfg.Render(select_coords, depths, is_train=False)
            #     pds[idx, S * cidx : S * (cidx + 1)] = rgb.cpu().numpy()
            # assert S * (cidx + 1) >= H * W
            
            for cidx in range(math.ceil(W * H / S)):
                l = S * cidx
                r = min(S * (cidx + 1), W*H)
                chunk = coords[:, l:r]
                # rgb, _ = cfg.fullmodel.module.eval_forward((chunk, depths,))
                rgb, _ = cfg.fullmodel((chunk, depths,))
                s = idx * coords.shape[0]
                e = min((idx + 1) * coords.shape[0], N)
                pds[s:e, l:r] = rgb.cpu().numpy() 

        pds = pds.reshape(N, H, W)
        
        if to_save: 
        ## log timing
            end_time = time.time()
            elapsed_time = (end_time - start_time) / len(dataloader)
            cfg.timing_file.write(f"{elapsed_time}\n")

            cfg.evaluation(pds)

def traineval(cfg, to_save=True):
    N, W, H, S = cfg.trainevalset.__len__(), cfg.trainevalset.W, cfg.trainevalset.H, cfg.bs_eval
    W, H = W//cfg.scale, H//cfg.scale
    print(f"traineval N: {N}, W: {W}, H: {H}, S: {S}")
    pds = np.zeros((N, W * H))
    dataloader = tqdm(cfg.trainevalloader)
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            dataloader.set_description(f'[TRNEVAL] : {idx}')
            coords, depths = batch
            # coords = coords.squeeze(0)
            # for cidx in range(math.ceil(W * H / S)):
            #     select_coords = coords[list(range(S * cidx, min(S * (cidx + 1), len(coords))))]
            #     rgb, _ = cfg.Render(select_coords, depths, is_train=False)
            #     pds[idx, S * cidx : S * (cidx + 1)] = rgb.cpu().numpy()
            # assert S * (cidx + 1) >= H * W
            
            for cidx in range(math.ceil(W * H / S)):
                l = S * cidx
                r = min(S * (cidx + 1), W*H)
                chunk = coords[:, l:r]
                # rgb, _ = cfg.fullmodel.module.eval_forward((chunk, depths,))
                rgb, _ = cfg.fullmodel((chunk, depths,))
                s = idx * coords.shape[0]
                e = min((idx + 1) * coords.shape[0], N)
                pds[s:e, l:r] = rgb.cpu().numpy()

        pds = pds.reshape(N, H, W)
        
        if to_save:
            end_time = time.time()
            elapsed_time = (end_time - start_time) / len(dataloader)
            cfg.timing_file.write(f'{elapsed_time:.4f}\n')
            cfg.evaluation(pds, "traineval")


def eval_size(cfg, to_save=True):
    N, W, H, S = cfg.evalset.__len__(), cfg.evalset.W, cfg.evalset.H, cfg.bs_eval
    pds = np.zeros((N, W * H, 3))
    dataloader = tqdm(cfg.evalloader)
    cfg.fullmodel.module.first_cycle = False

    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            dataloader.set_description(f'[EVAL] : {idx}')
            coords, depths = batch
            # coords = coords.squeeze(0)
            # for cidx in range(math.ceil(W * H / S)):
            #     select_coords = coords[list(range(S * cidx, min(S * (cidx + 1), len(coords))))]
            #     rgb, _ = cfg.Render(select_coords, depths, is_train=False)
            #     pds[idx, S * cidx : S * (cidx + 1)] = rgb.cpu().numpy()
            # assert S * (cidx + 1) >= H * W
            
            for cidx in range(math.ceil(W * H / S)):
                l = S * cidx
                r = min(S * (cidx + 1), W*H)
                chunk = coords[:, l:r]
                # rgb, _ = cfg.fullmodel.module.eval_forward((chunk, depths,))
                dxdydz = cfg.fullmodel.module.eval_size((chunk, depths,))
                s = idx * coords.shape[0]
                e = min((idx + 1) * coords.shape[0], N)
                pds[s:e, l:r] = dxdydz.cpu().numpy()

        pds = pds.reshape(N, H, W, 3)
        
        if to_save: 
        ## log timing
            ## save the images as rgb images?
            result_path = os.path.join(cfg.result_path, "eval_size")
            os.makedirs(result_path, exist_ok=True)

            names = ["dx", "dy", "dz"]
            text_size = 40
            for i in range(10):
                ## save the graph as three heat map plots
                for j in range(3):
                    nm = names[j]
                    savepath = os.path.join(result_path, f'{str(i).zfill(len(str(N)))}_{nm}.png')
                    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
                    plt.imshow(pds[i, :, :, j], cmap="viridis")
                    plt.title(nm, fontsize=text_size)
                    cbar = plt.colorbar()
                    cbar.ax.tick_params(labelsize=text_size)  # Adjust the label size as needed
                    plt.axis('off')
                    plt.savefig(savepath)
                    plt.close()
                
                ## save three images as a row with the same scale and name as dxdydz
                savepath = os.path.join(result_path, f'{str(i).zfill(len(str(N)))}.png')
                plt.figure(figsize=(30, 10))  # Adjust the figure size as needed
                for j in range(3):
                    nm = names[j]
                    plt.subplot(1, 3, j+1)
                    plt.imshow(pds[i, :, :, j], cmap="viridis")
                    plt.title(nm, fontsize=text_size)
                    cbar = plt.colorbar()
                    cbar.ax.tick_params(labelsize=text_size)  # Adjust the label size as needed
                    plt.axis('off')
                plt.savefig(savepath)
                plt.close()

                ## concatenate the three images into one image
                savepath = os.path.join(result_path, f'{str(i).zfill(len(str(N)))}_row.png')
                plt.figure(figsize=(30, 10))  # Adjust the figure size as needed
                plt.imshow(np.concatenate([pds[i, :, :, 0], pds[i, :, :, 1], pds[i, :, :, 2]], axis=1), cmap="viridis")
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=text_size)  # Adjust the label size as needed
                plt.axis('on')
                plt.savefig(savepath)
                plt.close()

                ## save with gt images 
                savepath = os.path.join(result_path, f'{str(i).zfill(len(str(N)))}_gt.png')
                plt.figure(figsize=(40, 10))  # Adjust the figure size as needed
                for j in range(3):
                    nm = names[j]
                    plt.subplot(1, 4, j+2)
                    plt.imshow(pds[i, :, :, j], cmap="viridis")
                    plt.title(nm, fontsize=text_size)
                    cbar = plt.colorbar()
                    cbar.ax.tick_params(labelsize=text_size)  # Adjust the label size as needed
                    plt.axis('off')
                gt = cfg.evalset.getLabel()
                plt.subplot(1, 4, 1)
                plt.imshow(gt[i, :, :], cmap="viridis")
                plt.title("gt", fontsize=text_size)
                plt.axis('off')
                plt.savefig(savepath)
                plt.close()

                ## save cube volume at each point 
                savepath = os.path.join(result_path, f'{str(i).zfill(len(str(N)))}_cube.png')
                plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
                plt.imshow(pds[i, :, :, 0] * pds[i, :, :, 1] * pds[i, :, :, 2], cmap="viridis")
                plt.title("cube", fontsize=text_size)
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=text_size)  # Adjust the label size as needed
                plt.axis('off')
                plt.savefig(savepath)
                plt.close()

        
def test(cfg):
    N, W, H, S = cfg.testset.__len__(), int(cfg.cam_scale * cfg.testset.W), int(cfg.cam_scale * cfg.testset.H), cfg.bs_test
    pds = np.zeros((N, H * W))
    dataloader = tqdm(cfg.testloader)
    axis = reduce(lambda x1, x2 : str(x1) + str(x2), [int(axis) for axis in cfg.axis])
    zs, angles, scales = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            coords, depths, R, zpos, angle, scale = batch
            zpos, angle, scale = zpos.item(), int(angle.item()), scale.item()
            zs.append(zpos)
            angles.append(angle)
            scales.append(scale)
            dataloader.set_description(f'[TEST] pos : {zpos:.2f} | axis : {axis} | angle : {angle} | scale : {scale:.2f}x')
            coords, R = torch.squeeze(coords), torch.squeeze(R)
            flags = utils.judge_range(coords, R)
            for cidx in range(H * W // S + 1):
                select_inds = list(range(S * cidx, min(S * (cidx + 1), len(coords))))
                select_flags = flags[select_inds]
                valid_inds = torch.tensor(select_inds).long()[select_flags]
                select_coords = coords[valid_inds]
                if len(select_coords) > 0:
                    valid_inds = valid_inds.cpu().numpy()
                    rgb, _ = cfg.Render(select_coords, depths, is_train=False, R=R)
                    rgb.cpu().numpy()
                    pds[idx, valid_inds] = rgb.cpu().numpy()
        pds = np.clip(pds.reshape((N, H, W)), 0, 1)

    if cfg.save_map: 
        cfg.Save_test_map(pds, zs, angles, scales)

def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def main(rank, worldsize, args, port):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ddp_setup(rank, worldsize, port)
    globals()[args.mode](Cfg(args, rank))
    
    
if __name__ == '__main__':
    
    args = argParse()
    
    if not args.multi_gpu: 

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        seed = 12341

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        globals()[args.mode](Cfg(args))
    else:
        print("add a dummy change for git push")
        world_size = torch.cuda.device_count()
        port = random.randint(10000, 20000)
        # port = 12345
        mp.spawn(main, args=(world_size, args, port), nprocs=world_size)
