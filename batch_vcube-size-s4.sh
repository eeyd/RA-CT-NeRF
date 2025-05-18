#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
scale=4
base_dir="kits19/data"
config_file="configs/vcube-size.yaml"
cases=("case_00010" "case_00045" "case_00052" "case_00089" "case_00120" "case_00135" "case_00140" "case_00162" "case_00197" "case_00210" "case_00230" "case_00291" "case_00295")

for case in "${cases[@]}"; do
    file="$base_dir/$case/${case}.nii.gz"
    echo "Running RA-CT NeRF for $case"
    python run.py RA-CT_NeRFx$scale --cfg $config_file --scale $scale --mode train --file $file --save_map --resume --multi_gpu
done