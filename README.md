# AI-Car-Racing

## Overview
Autonomous driving in continuous-action environments remains a challenging problem in Reinforcement Learning. This project uses Gymnasium's `CarRacing-V3` environment as it provides a controlled, simulated platform where an agent must learn to navigate a procedurally generated racetrack. The agent's goal is to maximize cumulative reward while avoiding collisions. This project aims to explore and benchmark three well-established learning approaches: Deep Q-Learning (DQN), Proximal Policy Optimization (PPO) with Advantage Actor-Critic (A2C) framework, and Behavioral Cloning (BC).

## Trained Agent Driving Samples
### PPO + A2C
https://github.com/user-attachments/assets/1eb313a3-0ffc-4c31-9c02-7304a4b08654

### Behavioral Cloning
https://github.com/user-attachments/assets/552735fc-3661-440a-96aa-8924f06906e0

### Deep Q-Learning Network
https://github.com/user-attachments/assets/21486f1f-9d24-42d1-b023-fcc640e1017c

## Methods
We experimented with multiple Reinforcement Learning methods such as Behavioral Cloning, Deep Q-Learning (with Discrete Action Space) and Proximal Policy Optimization through Actor-Critic Framework (PPO-A2C).

## Results
According to the literature, we discovered that a mean reward of 800 or more is considered good. Our best performing DQN and PPO + A2C methods are able to achieve a mean reward greater than 800, as noted in the Results section. The Behavioral Cloning method is not too far behind, with a mean reward of 743.54, which is also close to the target of 800. 

The DQN method was the best-performing solution with a mean reward of 866.8 and a standard deviation of 44.8. However, this cannot be directly compared to other RL methods we tried, since it's only using a Discrete Action space. 

Amongst the methods using Continuous Action space, the PPO + A2C method performed the best with a mean reward of 829.35, but it showed a high standard deviation of 145.8. As noted from the performance of the Behavioral Cloning method, the dataset used for training appears to be promising in capturing effective expert demonstrations, although on its own, it does not achieve competitive performance. We believe that the PPO + A2C method can be combined with pretraining using the Behavioral Cloning (expert) dataset in future work to improve mean reward performance and reduce the high variance, potentially offering strong competition to DQN’s performance in the discrete action space.

## Setup Instructions

### Prerequisites (Linux / Ubuntu)
Install system build tools and the Python headers before creating the venv:
```bash
sudo apt update
sudo apt install -y build-essential swig python3.11-dev python3.11-venv
```
### Prerequisites (Windows)
Download swig and set up system path:
https://swig.org/Doc1.3/Windows.html#Windows_examples

Visual Studios Build Tools 2026:

### Initial Setup
https://visualstudio.microsoft.com/visual-cpp-build-tools/
(make sure to check the "MSVC v143 - VS 2022 C++ build tools" package)

### Download uv
Please download `uv` (Ultra-Violet) for Python Project Dependency Management: https://docs.astral.sh/uv/getting-started/installation/#installation-methods

### Initializing a uv virtual env
Run following commands by navigating to the project directory:
```bash
cd /path/to/your/project
uv sync
```

### Activating the virtual env
In the same project directory, execute the following (if virtual env is not already active):
```bash
source .venv/bin/activate
```
### Windows
```
.\.venv\Scripts\Activate.ps1    
```

### Adding any Libraries / Dependencies
To add any new dependencies (libraries):
```bash
uv add <library_name>
```

## Playing the Car Racing Game Manually
Please run the following command from the project directory:

For MacOS / Linux:
```bash
uv run .venv/lib/python3.11/site-packages/gymnasium/envs/box2d/car_racing.py
```

For Windows:
```bash
uv run .venv/lib/site-packages/gymnasium/envs/box2d/car_racing.py
```
