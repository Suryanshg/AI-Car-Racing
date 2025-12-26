# AI-Car-Racing

## Overview
TODO

## Methods
TODO

## Results
TODO


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
