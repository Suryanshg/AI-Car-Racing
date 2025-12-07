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
# Load modules
# -----------------------------
module load python/3.12.10   # Use Python version compatible with uv
module load cuda

# -----------------------------
# Create logs folder
# -----------------------------
mkdir -p logs

# -----------------------------
# Create or activate uv venv
# -----------------------------
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating uv virtual environment..."
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# -----------------------------
# Install/update dependencies
# -----------------------------
echo "Syncing dependencies with uv..."
uv sync

# -----------------------------
# Run PPO training
# -----------------------------
uv run python main_ppo.py --train_ppo