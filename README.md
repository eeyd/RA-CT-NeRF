# RA-CT NeRF

## Abstract

In this work, we introduce Region Adaptive CT NeRF (RA-CT NeRF), a novel framework for arbitrary-scale super-resolution of cross-sectional CT images. The proposed method addresses the challenges of anisotropic imaging by incorporating a region-adaptive sampling strategy that dynamically adjusts cube sizes based on local feature complexity. This adaptive approach ensures efficient representation of both smooth and detailed regions, significantly improving convergence and reconstruction quality. Furthermore, we implement an alternating training scheme to optimize the color and cube size networks, enhancing stability and performance. Extensive experiments demonstrate that RA-CT NeRF outperforms state-of-the-art methods, achieving superior results across various scaling factors while preserving fine details and mitigating oversmoothing. 

## 1) Get start

* Python 3.9.x
* CUDA 11.1 or *higher*
* Torch 1.8.0 or *higher*

**Create a python env using conda**
```bash
conda create -n RA-CT NeRF python=3.9 -y
```

**Install the required libraries**
```bash
bash setup.sh
```

**Additional packages**
exact verisons are listed in enc_ref.txt

**[option] Install FFmpeg**
```bash
apt install ffmpeg -y
```

## 2) Download kits-19 dataset
please visit **[kits19](https://github.com/neheller/kits19.git)**. We thank them for their wonderful work and code release. Put kits19 under root directory as such `RA-CT-NeRF/kits19/data`. And remame `case_00010/imaging.nii.gz` to `case_00010/case_00010.nii.gz` for correct path specification. 


## 3) Training RA-CT NeRF for medical volumes
```bash
bash batch_method_name.sh
```
scale can be set in the bash script. Other parameters are in configs folder, please refer to the sample yaml files. `--mode train` for training. `--mode eval_size` for outputting cube size visualisation.

## Acknowledgement 

We build our project based on **[CuNeRF](https://github.com/NarcissusEx/CuNeRF.git)**. We thank them for their wonderful work and code release.