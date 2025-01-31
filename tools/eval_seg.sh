#!/bin/sh

#SBATCH --job-name=cutsegeval
#SBATCH --output=seg/logs/cutsegeval-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=seg/logs/cutsegeval-%A.err  # Standard error of the script
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

# load python module
ml miniconda3

# activate corresponding environment
conda deactivate
conda activate cutseg

# link to the dataset folder, model weights and the config file.
export DETECTRON2_DATASETS=../dataset
model_weights="seg/output/isic2016-sup/model_0009999.pth"
config_file="seg/model_zoo/configs/cutseg-R50-FPN.yaml"
num_gpus=1

test_dataset='isic2016_test'
echo "========== evaluating ${test_dataset} =========="
python seg/train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only \
  --opts MODEL.WEIGHTS ${model_weights}
