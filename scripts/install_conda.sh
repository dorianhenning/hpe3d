#!/usr/bin/env bash

export CONDA_ENV_NAME=hpe3d-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
conda install libspatialindex==1.9.3  # trimesh dependency
pip install -r requirements.txt

pip install -e .

# Install detectron2 for bounding boxes
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# make data folder and download model checkpoint & extras
mkdir -p data/smpl

# Model checkpoint from SPIN
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data

# Additional data (only extract necessary structs)
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz
tar -xvf data.tar.gz data/J_regressor_extra.npy data/smpl_mean_params.npz
rm data.tar.gz

# find and copy the newest smpl basicModel file to SMPL_NEUTRAL
find ~ -name "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" | head -1 | xargs -I {} cp {} ./data/smpl/SMPL_NEUTRAL.pkl
