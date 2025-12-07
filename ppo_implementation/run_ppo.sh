#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "PPO-A2C"
#SBATCH -p academic
#SBATCH -A cs551
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -o logs/logs.out
#SBATCH -e logs/logs.out

# -----------------------------
# Environment Setup
# -----------------------------
module load swig/4.3.0
module load gcc/13.3.0
module load python/3.11.6
module load cuda

# Ensure GCC libraries are in path
GCC_LIB_PATH=$(dirname $(dirname $(which g++)))/lib64
export LD_LIBRARY_PATH=$GCC_LIB_PATH:$LD_LIBRARY_PATH

# Create logs folder
mkdir -p logs

# -----------------------------
# Create or activate venv
# -----------------------------
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

# -----------------------------
# Run PPO training
# -----------------------------
python -u main_ppo.py --train_ppo