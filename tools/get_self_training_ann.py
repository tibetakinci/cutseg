#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import tqdm
import torch
import datetime
import argparse
import pycocotools.mask as cocomask
from detectron2.utils.file_io import PathManager
from PIL import Image

INFO = {
    "description": "ImageNet-1K: Self-train",
    "url": "",
    "version": "1.0",
    "year": 2022,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'lesion',
        'supercategory': 'foreground',
    },
]

new_dict_filtered = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

category_info = {
    "is_crowd": 0,
    "id": 1
}


def segmToRLE(segm, h, w):
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask

def cocosegm2mask(segm, h, w):
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask

def BatchIoU(masks1, masks2):
    n1, n2 = masks1.size()[0], masks2.size()[0]
    masks1, masks2 = (masks1>0.5).to(torch.bool), (masks2>0.5).to(torch.bool)
    masks1_ = masks1[:,None,:,:,].expand(-1, n2, -1, -1)
    masks2_ = masks2[None,:,:,:,].expand(n1, -1, -1, -1)

    intersection = torch.sum(masks1_ * (masks1_ == masks2_), dim=[-1, -2])
    union = torch.sum(masks1_ + masks2_, dim=[-1, -2])
    ious = intersection.to(torch.float) / union
    return ious

def IoU(mask1, mask2):
    # Convert to boolean tensors where 1s are True and 0s are False
    mask1 = mask1 == 1
    mask2 = mask2 == 1
    
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    
    if union == 0:
        iou = 0
    else:
        iou = intersection / union
    
    return iou

if __name__ == "__main__":
    # load model arguments
    parser = argparse.ArgumentParser(description='Generate json files for the self-training')
    parser.add_argument('--new-pred', type=str, 
                        default='output/inference/coco_instances_results.json',
                        help='Path to model predictions')
    parser.add_argument('--prev-ann', type=str, 
                        default='DETECTRON2_DATASETS/imagenet/annotations/cutler_imagenet1k_train.json',
                        help='Path to annotations in the previous round')
    parser.add_argument('--save-path', type=str,
                        default='DETECTRON2_DATASETS/imagenet/annotations/cutler_imagenet1k_train_r1.json',
                        help='Path to save the generated annotation file')
    args = parser.parse_args()

    # load model predictions
    new_pred = args.new_pred
    with PathManager.open(new_pred, "r") as f:
        predictions = json.load(f)

    # filter out low-confidence model predictions
    pred_image_to_anns = {}
    for id, ann in enumerate(predictions):
        pred_image_to_anns[id] = ann

    # load psedu-masks used by the previous round
    pseudo_ann_dict = json.load(open(args.prev_ann))
    pseudo_image_list = pseudo_ann_dict['images']
    pseudo_annotations = pseudo_ann_dict['annotations']

    pseudo_image_to_anns = {}
    for id, ann in enumerate(pseudo_annotations):
        pseudo_image_to_anns[id] = ann

    # merge model predictions and the json file used by the previous round.
    merged_anns = []

    for k, anns_pseudo in tqdm.tqdm(pseudo_image_to_anns.items()):
        segm = anns_pseudo['segmentation']
        mask = cocosegm2mask(segm, segm['size'][0], segm['size'][1])
        pseudo_mask = torch.from_numpy(mask)
        try:
            index=2*k+1
            anns_pred = pred_image_to_anns[index]
        except:
            merged_anns += anns_pseudo
            continue

        segm = anns_pred['segmentation']
        mask = cocosegm2mask(segm, segm['size'][0], segm['size'][1])
        pred_mask = torch.from_numpy(mask)

        iou_fg = IoU(pseudo_mask, pred_mask)
        if iou_fg < 0.5:
            anns_pseudo['segmentation'] = anns_pred['segmentation']

        merged_anns += anns_pseudo

    for key in pred_image_to_anns:
        if key in pseudo_image_to_anns:
            continue
        else:
            merged_anns += pred_image_to_anns[key]

    # re-generate annotation id
    ann_id = 1
    for ann in merged_anns:
        ann['id'] = ann_id
        #ann['area'] = ann['bbox'][-1] * ann['bbox'][-2]
        ann['iscrowd'] = 0
        ann['width'] = ann['segmentation']['size'][0]
        ann['height'] = ann['segmentation']['size'][1]
        ann_id += 1

    new_dict_filtered['images'] = pseudo_image_list
    new_dict_filtered['annotations'] = merged_anns

    # save annotation file
    with open(args.save_path, 'w') as output_json_file:
        json.dump(new_dict_filtered, output_json_file)
    print("Done: {} images; {} anns.".format(len(new_dict_filtered['images']), len(new_dict_filtered['annotations'])))