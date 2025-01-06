#!/bin/bash

#SBATCH --gres=gpu:01
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-user=martin.gillis@dal.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/hmgillis/projects/rrg-ttt/hmgillis/cathepsin/slurm/logs/slurm-%j.log

# define current project
PROJECT=/home/hmgillis/projects/rrg-ttt/hmgillis/cathepsin

# load modules
module purge
module load python/3.11.5 cuda/12.2 cudnn/8.9.5.29

# create and activate virtual environment
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# install python packages
pip install --no-index --upgrade pip

pip install --no-index torch torchvision
pip install --no-index transformers
pip install --no-index matplotlib
pip install --no-index pandas
pip install --no-index pytest
pip install --no-index toml

# create data directory and extract files
mkdir $SLURM_TMPDIR/datasets
tar -xzf /home/hmgillis/projects/rrg-ttt/hmgillis/datasets/alentic/mae/datasets.tar.gz -C $SLURM_TMPDIR/datasets

# list contents of data directory
ls $SLURM_TMPDIR/datasets

# run script
python $PROJECT/vitmae.py --config $PROJECT/configs/vitmae.toml

# deactivate virtual environment
deactivate
