#!/bin/bash
# Five-minute SLURM job: does the conda env see the GPU?
#   sbatch -A <your-account> -p <gpu-partition> gpu_check.sbatch
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4
#SBATCH --mem=16G
#SBATCH -t 00:05:00
#SBATCH -J rlhf-gpu-check
#SBATCH -o out/gpu_check.log
ENV="${ENV:-/scratch/gilbreth/$USER/envs/rlhf}"
hostname
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
"$ENV/bin/python" - <<PY
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", (x @ x).sum().item() != 0)
print(torch.cuda.get_device_name(0))
PY
