#!/bin/bash

# define project root
PROJECT=/home/hmgillis/projects/rrg-ttt/hmgillis/cathepsin

# submit slurm job (cannot run sbatch in home directory)
cd $PROJECT && sbatch $PROJECT/slurm/vitmae.sh
