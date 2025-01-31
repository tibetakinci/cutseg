#!/bin/sh

#SBATCH --job-name=cutsegtrain
#SBATCH --output=seg/logs/cutsegtrain-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=seg/logs/cutsegtrain-%A.err  # Standard error of the script
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=96G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)
#SBATCH --nodelist=corellia

# load python module
ml miniconda3

# activate corresponding environment
conda deactivate
conda activate cutseg

# run the program
python seg/train_net.py --config-file seg/model_zoo/configs/cutseg-R50-FPN.yaml --num-gpus 1 --resume