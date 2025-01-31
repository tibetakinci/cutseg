import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import datetime
import PIL
import PIL.Image as Image
import torch
from scipy import ndimage
import json
import matplotlib.pyplot as plt

# modfied by Xudong Wang based on third_party/TokenCut
sys.path.append('../third_party')
from TokenCut.unsupervised_saliency_detection import metric

from feat import ViTFeat
from crf import densecrf
from cut import *

    

# necessary info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2023,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.now(datetime.UTC).isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'lesion',
        'supercategory': 'foreground',
    },
]

category_info = {
    "is_crowd": 0,
    "id": 1
}


class Predictor():
    def __init__(self, args):
        """Initialize"""
        if args.vit_model == 'dinov2':
            args.patch_size = 14
            args.vit_feat = 'q'
            args.fixed_size = 476
        elif args.vit_model == 'sammed':
            args.fixed_size = 256
        elif args.vit_model == 'sam':
            args.fixed_size = 1024
        elif args.vit_model == 'pretrained':
            if args.pretrained_dir is None:
                raise ValueError('Pretrained directory is None')

        self.backbone = ViTFeat(args.vit_arch, args.vit_feat, args.patch_size, args.vit_model, args.pretrained_dir)
        self.backbone.eval()
        if not args.cpu:
            self.backbone.cuda()

        self.args = args


    def evaluate(self, pred_mask, gt_mask):
        """Evaluate prediction"""
        pred_mask[pred_mask == 255] = 1
        gt_mask = np.copy(gt_mask)
        gt_mask[gt_mask == 255] = 1

        intersection_fg = np.sum((pred_mask == 1) & (gt_mask == 1))
        intersection_bg = np.sum((pred_mask == 0) & (gt_mask == 0))
        uni_fg = np.sum((pred_mask == 1) | (gt_mask == 1))
        uni_bg = np.sum((pred_mask == 0) | (gt_mask == 0))
        union = np.sum(pred_mask) + np.sum(gt_mask)

        correct = np.sum(pred_mask == gt_mask)
        total = gt_mask.size
        
        dice = 2.0 * intersection_fg / union
        accuracy = correct / total
        iou_fg = intersection_fg / uni_fg
        iou_bg = intersection_bg / uni_bg

        return {"iou": (iou_fg + iou_bg) / 2, "dice": dice, "acc": accuracy}

    
    def predict_single(self, img_path):
        """Create pseudo-mask for given image"""
        # get pseudo-masks for each image using MaskCut
        try:
            bipartition, _, I_new = cut(img_path, self.backbone, args.patch_size, \
                fixed_size=args.fixed_size, cpu=args.cpu, laplacian=args.laplacian)
        except:
            print(f'Skipping {img_path.split('/')[-1]}')
            return None, None

        I = Image.open(img_path).convert('RGB')
        width, height = I.size

        # post-process pesudo-masks with CRF
        pseudo_mask = densecrf(np.array(I_new), bipartition)
        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

        # filter out the mask that have a very different pseudo-mask after the CRF
        mask1 = torch.from_numpy(bipartition)
        mask2 = torch.from_numpy(pseudo_mask)
        if not args.cpu: 
            mask1 = mask1.cuda()
            mask2 = mask2.cuda()
        if metric.IoU(mask1, mask2) < 0.3:  #0.5
            pseudo_mask = pseudo_mask * -1

        # construct binary pseudo-masks
        pseudo_mask[pseudo_mask < 0] = 0
        pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
        pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))
        pseudo_mask = pseudo_mask.astype(np.uint8)
        upper = np.max(pseudo_mask)
        lower = np.min(pseudo_mask)
        thresh = upper / 2.0
        pseudo_mask[pseudo_mask > thresh] = upper
        pseudo_mask[pseudo_mask <= thresh] = lower

        return pseudo_mask, I


    def predict(self):
        """Refine pseudo-mask, evaluate and export"""
        img_folders = sorted(os.listdir(args.imgs_path))
        gt_folders = sorted(os.listdir(args.gt_path))
        
        start_idx = max(args.job_index*args.num_folder_per_job, 0)
        end_idx = min((args.job_index+1)*args.num_folder_per_job, len(img_folders))

        image_id, segmentation_id = 1, 1
        results = {"dice": [], "miou": [], "acc": []}
        fail_cases = []
        output = {
                "info": INFO,
                "licenses": LICENSES,
                "categories": CATEGORIES,
                "images": [],
                "annotations": []}

        for img_name, gt_name in tqdm(zip(img_folders[:1], gt_folders[:1]), total=1): #len(img_folders)  -  [start_idx:end_idx]
            # get image and gt path
            img_dir = os.path.join(args.imgs_path, img_name)
            gt_dir = os.path.join(args.gt_path, gt_name)

            pseudo_mask, I = self.predict_single(img_dir)
            if pseudo_mask is None:
                continue
            
            gt = Image.open(gt_dir)
            gt = np.asarray(gt)

            result = self.evaluate(pseudo_mask, gt)

            results["miou"].append(result["iou"])
            results["acc"].append(result["acc"])
            results["dice"].append(result["dice"])
            if result["dice"] < 0.5:
                fail_cases.append(img_name)

            height, width = I.size
            # create coco-style image info
            image_info = create_image_info_coco(
                image_id, "{}".format(img_name), (height, width, 3))    
            output["images"].append(image_info)

            # create coco-style annotation info
            annotation_info = create_annotation_info_coco(
                segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
            if annotation_info is not None:
                output["annotations"].append(annotation_info)
                segmentation_id += 1
            image_id += 1

        print(f"Dice: {np.mean(results["dice"])}")
        print(f"Acc: {np.mean(results["acc"])}")
        print(f"mIoU: {np.nanmean(results["miou"])}")
        print(len(fail_cases))
        print(fail_cases)

        # save annotations
        if True: #len(img_folders) == args.num_folder_per_job and args.job_index == 0:
            json_name = '{}/{}_train_fixsize{}_lap{}_{}{}.json'.format(args.ann_path, args.dataset_name, args.fixed_size, args.laplacian, args.vit_arch, args.patch_size)
        else:
            json_name = '{}/{}_train_fixsize{}_lap{}_{}{}_{}_{}.json'.format(args.ann_path, args.dataset_name, args.fixed_size, args.laplacian, args.vit_arch, args.patch_size, start_idx, end_idx)
        with open(json_name, 'w') as output_json_file:
            json.dump(output, output_json_file)
        print(f'dumping {json_name}')
        print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))


    def demo(self):
        if not os.path.exists(args.img_path):
            raise ValueError(f"img does not exist {args.img_path}")

        img_name = args.img_path.split('/')[-1]
        pseudo_mask, _ = self.predict_single(args.img_path)
        
        gt_dir = os.path.join(args.gt_path, img_name.replace('.jpg', '_Segmentation.png'))
        gt = Image.open(gt_dir)
        gt = np.asarray(gt)

        result = self.evaluate(pseudo_mask, gt)
        print(f"Dice: {result["dice"]}")
        print(f"Acc: {result["acc"]}")
        print(f"mIoU: {result["iou"]}")

        pseudo_mask = (pseudo_mask * 255).astype(np.uint8)
        img = Image.fromarray(pseudo_mask)
        out_dir = os.path.join(args.output_path, img_name.replace('.jpg','_mask.png'))
        img.save(out_dir)
        print(f"img saved to {out_dir}")


def setup():
    """Get arguments"""
    parser = argparse.ArgumentParser('Cut script')

    # model arguments
    parser.add_argument('--vit-model', type=str, default='dino', choices=['dino', 'dinov2', 'sammed', 'sam', 'sam2', 'pretrain'], help='which model')
    parser.add_argument('--vit-arch', type=str, default='base', choices=['base', 'small', 'large', 'huge'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8, 14], help='patch size')
    parser.add_argument('--laplacian', type=str, default='comb', choices=['comb', 'norm', 'rw'], help='how to calculate laplacian')
    parser.add_argument('--pretrained-dir', type=str, default='pretrain/dino_s16.pth', help='pretrained backbone directory')

    # data arguments
    parser.add_argument('--dataset-name', type=str, default="isic2016", help='dataset name')
    parser.add_argument('--imgs-path', type=str, default="../../dataset/ISIC2016/train/imgs", help='path to the images')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')
    parser.add_argument('--gt-path', type=str, default="../../dataset/ISIC2016/train/orig_gt", help='path to the ground truth')
    parser.add_argument('--output-path', type=str, default="./output", help='output path to save mask')
    parser.add_argument('--ann-path', type=str, default="./annotations", help='output path to save annotations')

    # mode arguments
    parser.add_argument('--demo', action='store_true', help='demo')

    # saving arguments
    parser.add_argument('--eval', action='store_false', help='evaluate results')
    parser.add_argument('--save-mask', action='store_false', help='save masks as png file')
    parser.add_argument('--save-ann', action='store_false', help='save annotations as json file')

    # additional arguments
    parser.add_argument('--num-folder-per-job', type=int, default=1000, help='the number of folders each job processes')
    parser.add_argument('--job-index', type=int, default=0, help='the index of the job (for imagenet: in the range of 0 to 1000/args.num_folder_per_job-1)')
    parser.add_argument('--fixed-size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--cpu', action='store_true', help='use cpu')

    return parser.parse_args()

if __name__ == "__main__":
    args = setup()
    predictor = Predictor(args)

    if args.demo:
        predictor.demo()
    else:
        predictor.predict()