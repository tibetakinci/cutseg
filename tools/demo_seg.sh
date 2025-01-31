#!/bin/sh

#SBATCH --job-name=cutsegdemo
#SBATCH --output=seg/logs/cutsegdemo-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=seg/logs/cutsegdemo-%A.err  # Standard error of the script
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

# load python module
ml miniconda3

# activate corresponding environment
conda deactivate
conda activate cutseg

python seg/demo/demo.py --config-file model_zoo/configs/cutseg-R50-FPN.yaml \
    --input ../dataset/ISIC2016/test/imgs/ISIC_0010229.jpg \
    --output results/isic16-r50-norm/ \
    --opts MODEL.WEIGHTS output/isic2016-r50-norm-dino/model_0002999.pth