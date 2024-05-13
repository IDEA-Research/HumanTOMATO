#!/bin/bash
#SBATCH -J X-TMR
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:hgx:4
#SBATCH --mem 300GB
#SBATCH --qos=preemptive

source activate temos

# python -m train --cfg configs/configs_temos/MotionX-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
# python -m train --cfg configs/configs_temos/UniMocap-TMR.yaml --cfg_assets configs/assets.yaml --nodebug
python -m train --cfg configs/configs_temos/H3D-TMR.yaml --cfg_assets configs/assets.yaml --nodebug


# find ./  -type d -name "__pycache__" -exec rm -rf {} +