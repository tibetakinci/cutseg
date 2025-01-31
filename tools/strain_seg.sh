#!/bin/sh

#SBATCH --job-name=cutsegselftrain
#SBATCH --output=seg/logs/cutsegselftrain-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=seg/logs/cutsegselftrain-%A.err  # Standard error of the script
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=6  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=50G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

# load python module
ml miniconda3

# activate corresponding environment
conda deactivate
conda activate cutseg

# variables
num_gpus=1
config_file=seg/model_zoo/configs/cutseg-R50-FPN.yaml
test_dataset=isic2016_train_comb
weights=seg/output/isic2016-r50-comb-dino/model_0002999.pth
output_dir=seg/output/isic2016-strain

:'
# get model predictions
python seg/train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only \
  --opts MODEL.WEIGHTS ${weights}  OUTPUT_DIR ${output_dir}
'

# variables
model_preds=seg/output/isic2016-strain/inference/sem_seg_predictions.json 
prev_ann=../dataset/ISIC2016/train/annotation/isic2016_trial_train_fixsize480_lapcomb_vit_base16.json
save_path=../dataset/ISIC2016/train/annotation/cutseg_isic2016_train_r1.json

# export self training annotations
python tools/get_self_training_ann.py --new-pred ${model_preds} \
  --prev-ann ${prev_ann} \
  --save-path ${save_path}

# variables
train_dataset=isic2016_train_r1
output_dir=seg/output/isic2016-strain
:'
# self training loop
python seg/train_net.py --config-file ${config_file} \
  --train-dataset ${train_dataset} \
  --opts OUTPUT_DIR ${output_dir}
'