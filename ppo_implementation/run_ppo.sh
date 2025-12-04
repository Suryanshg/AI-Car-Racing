#!/bin/bash
#SBATCH -N 1                    
#SBATCH -n 16                    
#SBATCH --mem=16g                
#SBATCH -J "PPO-A2C-action-repeat"    

#SBATCH -p academic
#SBATCH -A cs551

##SBATCH -p short

#SBATCH -t 48:00:00             
#SBATCH --gres=gpu:1

#SBATCH -C A30
##SBATCH -C A100

#SBATCH -o logs_action_repeat.out 
#SBATCH -e logs_action_repeat.out

module load swig/4.3.0
module load gcc/13.3.0
module load python/3.11.6              
module load cuda

GCC_LIB_PATH=$(dirname $(dirname $(which g++)))/lib64
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

source .venv/bin/activate
python -u main_ppo.py --train_ppo
