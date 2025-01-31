#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage
import matplotlib.pyplot as plt

from feat import ViTFeat
sys.path.append('../third_party')
sys.path.append('third_party/')
from TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from cut import cut

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MaskCut Demo')
    # default arguments
    parser.add_argument('--out-dir', type=str, default='demo/', help='output directory')
    parser.add_argument('--vit-model', type=str, default='dino', choices=['dino', 'dinov2', 'sammed', 'sam', 'pretrain'], help='which model')
    parser.add_argument('--vit-arch', type=str, default='base', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')
    parser.add_argument('--laplacian', type=str, default='comb', choices=['comb', 'norm', 'rw', 'hodge'], help='how to compute laplacian')
    parser.add_argument('--pretrained-dir', type=str, default='pretrain/dino_s16.pth', help='pretrained backbone directory')
    parser.add_argument('--clahe', action='store_true', help='use CLAHE data augmentation method')
    parser.add_argument('--hair', action='store_true', help='use hair removal data augmentation method')

    # additional arguments
    parser.add_argument('--fixed-size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain-path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--cpu', action='store_true', help='use cpu')

    args = parser.parse_args()

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
    
    backbone = ViTFeat(args.vit_arch, args.vit_feat, args.patch_size, args.vit_model, args.pretrained_dir)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    bipartition, second_eig, I_new = cut(args.img_path, backbone, args.patch_size, \
                fixed_size=args.fixed_size, cpu=args.cpu, laplacian=args.laplacian, clahe=args.clahe, hair=args.hair)    
    I = Image.open(args.img_path).convert('RGB')
    width, height = I.size
    pseudo_mask_list = []
        
    # post-process pesudo-masks with CRF
    pseudo_mask = densecrf(np.array(I_new), bipartition)
    pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)
    
    # filter out the mask that have a very different pseudo-mask after the CRF
    mask1 = torch.from_numpy(bipartition)
    mask2 = torch.from_numpy(pseudo_mask)
    if not args.cpu: 
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
    if metric.IoU(mask1, mask2) < 0.3:
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

    os.makedirs(args.out_dir, exist_ok=True)
    img_name = args.img_path.split('/')[-1].split('.')[0]
    im = Image.fromarray(pseudo_mask)
    im.save(os.path.join(args.out_dir, f"{img_name}_psmask.jpg"))