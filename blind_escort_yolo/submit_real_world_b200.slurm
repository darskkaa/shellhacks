#!/bin/bash
#SBATCH --job-name=yolo-real-b200
#SBATCH --partition=hpg-b200
#SBATCH --account=cqu.fgcu
#SBATCH --qos=cqu.fgcu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15000M
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --output=/blue/cqu.fgcu/speppers5329.fgcu/waymo_curbrisk/logs/yolo_real_%j.log
#SBATCH --error=/blue/cqu.fgcu/speppers5329.fgcu/waymo_curbrisk/logs/yolo_real_%j.err

echo "=== UF HiPerGator B200 Compute Node Allocation for Real-World YOLO ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /blue/cqu.fgcu/speppers5329.fgcu/waymo_curbrisk
mkdir -p logs checkpoints datasets runs

module load pytorch/2.8.0
export PYTHONPATH="$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH"

# Run fine-tuning on Blackwell B200 with real-world 24k assistive vision dataset
python3 train_real_world_b200.py

echo "=== Real-World YOLO Training Job Completed Successfully ==="
