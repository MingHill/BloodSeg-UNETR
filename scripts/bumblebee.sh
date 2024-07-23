#! /usr/bin/zsh

# load environment
source .pyenv/versions/default/bin/activate

# define project root (bumblebee)
PROJECT="/home/hmgillis/projects/python/cathepsin"

# run python script
cd $PROJECT && python $PROJECT/vitmae.py --config configs/vitmae.toml
#cd $PROJECT && python $PROJECT/vitmae-cifar.py --config configs/vitmae-cifar.toml
