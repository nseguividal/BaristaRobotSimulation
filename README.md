# BaristaRobotSimulation
Project for the course Planning and Decision Making followed at TU Delft, where we implement a planning algorithm for a barista robot. 


## Installation (only once)

# 1. Open terminal and go to repo
cd ~/PDM_project/BaristaRobotSimulation/

## 2. Deactivate any active virtualenvs
conda deactivate || true

## 3. Create + activate a conda environment (one-time)
conda create -n gym_env python=3.11 -y
conda activate gym_env

## 4. Install PyBullet + core deps from conda-forge
conda install -c conda-forge pybullet gymnasium numpy pyopengl -y

## 5. Install this repo as an editable package so "urdfenvs" is importable
pip install -e .

## 6. Verify pybullet is available
python -c "import pybullet as p; print('pybullet', p.__version__)"

## 7. Run the example
python tests/albert.py

## 8. When done, press Ctrl+C to stop and then:
conda deactivate


## To run a file 

# 1. Open terminal and go to repo
cd ~/PDM_project/BaristaRobotSimulation/

# 2. Activate gym enviroment 
conda activate gym_env

# 3. Run files
python tests/barista_scenario.py




### Extras
To see all enviroments: conda info --envs

URDF file of the albert robot taken from this git repo: https://github.com/maxspahn/gym_envs_urdf

Kill pybullet: control + C 

To move around: press control 