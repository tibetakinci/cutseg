#!/bin/sh

#SBATCH --job-name=cut
#SBATCH --output=cut/logs/cut-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cut/logs/cut-%A.err  # Standard error of the script
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

# activate corresponding environment
conda deactivate
conda activate cutseg

# running parameters:
laplacian=comb
patch_size=16

# dataset parameters:
## ISIC2016
: '
imgs_path=../dataset/ISIC2016/train/imgs
gt_path=../dataset/ISIC2016/train/orig_gt
dataset_name=isic2016
output_dir=../dataset/ISIC2016/train/annotation
gt_suffix=_Segmentation

## ISIC2017
imgs_path=../dataset/ISIC2017/train/imgs
gt_path=../dataset/ISIC2017/train/label
dataset_name=isic2017
output_dir=../dataset/ISIC2017/train/annotation
gt_suffix=_segmentation
'
## ISIC2018
imgs_path=../dataset/ISIC2018/train/imgs
gt_path=../dataset/ISIC2018/train/label
dataset_name=isic2018
output_dir=../dataset/ISIC2018/train/annotation
gt_suffix=_segmentation
: '
## ISIC2019 - REMOVE --eval, --gt-path, --gt-suffix
imgs_path=../dataset/ISIC2019/train/imgs
dataset_name=isic2019
output_dir=../dataset/ISIC2019/train/annotation

## ISIC2020 - REMOVE --eval, --gt-path, --gt-suffix
imgs_path=../dataset/ISIC2020/train/imgs
dataset_name=isic2020
output_dir=../dataset/ISIC2020/train/annotation

## 7point - REMOVE --eval, --gt-path, --gt-suffix
imgs_path=../dataset/7point/cropped_imgs
dataset_name=7point
output_dir=../dataset/7point/annotation

## PH2
imgs_path=../dataset/PH2/train/imgs
gt_path=../dataset/PH2/train/label
dataset_name=ph2
output_dir=../dataset/PH2/train/annotation
gt_suffix=_lesion
'

# run the program
python cut/cut.py --imgs-path ${imgs_path} --gt-path ${gt_path} --dataset-name ${dataset_name} --out-dir ${output_dir} --gt-suffix ${gt_suffix} --laplacian ${laplacian} --patch_size ${patch_size}