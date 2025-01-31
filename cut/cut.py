#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
from tqdm import tqdm
import re
import datetime
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
import json
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, coo_matrix
from scipy.sparse.linalg import eigsh
import itertools
import sklearn.neighbors
from scipy.spatial.distance import cdist
import math
import toponetx as tnx
from skimage.measure import find_contours
from scipy.spatial import distance
import cv2
from itertools import combinations

sys.path.append('../third_party')
sys.path.append('third_party/')
from TokenCut.unsupervised_saliency_detection import utils, metric
from TokenCut.unsupervised_saliency_detection.object_discovery import detect_box

sys.path.append('../utils')
sys.path.append('utils/')
from draw import draw_legth_width, longest_length_and_width, plot_contour_length_width
from data_aug import do_aug

from feat import ViTFeat
from crf import densecrf


__all__ = ["cut", "create_image_info_coco", "create_annotation_info_coco"]


# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])


def metrics(pred_mask, gt_mask):
    intersection_fg = np.sum((pred_mask == 1) & (gt_mask == 1))
    intersection_bg = np.sum((pred_mask == 0) & (gt_mask == 0))
    uni_fg = np.sum((pred_mask == 1) | (gt_mask == 1))
    uni_bg = np.sum((pred_mask == 0) | (gt_mask == 0))
    union = np.sum(pred_mask) + np.sum(gt_mask)

    correct = np.sum(pred_mask == gt_mask)
    total = gt_mask.size
    accuracy = correct / total
    iou_fg = intersection_fg / uni_fg
    iou_bg = intersection_bg / uni_bg

    #union_fg = np.sum(pred_mask == 1) + np.sum(gt_mask == 1)
    #union_bg = np.sum(pred_mask == 0) + np.sum(gt_mask == 0)
    #dice_fg = (2.0 * intersection_fg) / union_fg
    #dice_bg = (2.0 * intersection_bg) / union_bg
    #dice = (dice_fg + dice_bg) / 2

    intersect = np.sum(pred_mask * gt_mask)
    total_sum = np.sum(pred_mask) + np.sum(gt_mask)
    dice = np.mean(2*intersect/total_sum)

    return {"iou": (iou_fg + iou_bg) / 2, "dice": dice, "acc": accuracy}

def get_diagonal(W, threshold: float = 1e-12):
    """Gets the diagonal sum of a sparse matrix"""
    try:
        from pymatting.util.util import row_sum
    except:
        raise ImportError(
            'Please install pymatting to compute the diagonal sums:\n'
            'pip3 install pymatting'
        )

    D = row_sum(W)
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = diags(D)
    return D

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )

    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []
    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)
    
    # This is our affinity matrix
    W = csr_matrix((coo_data, (ij, ji)), (n, n))
    return W

def compute_laplacian(W, lap="comb", s=1.0):
    D = np.array(get_diagonal(W).todense())
    if lap == "comb":
        # combinatorial laplacian
        L = D - W
    elif lap == "norm":
        # normalized laplacian
        D_inv_sqrt = np.linalg.pinv(np.sqrt(D))
        I = np.eye(W.shape[0])
        L = s*I - D_inv_sqrt @ W @ D_inv_sqrt
    elif lap == "rw":
        # random walk laplacian
        D_inv = np.linalg.pinv(D)
        I = np.eye(W.shape[0])
        L = s*I - D_inv @ W
    else:
        raise NotImplementedError()
    
    return L.astype(np.float32), D.astype(np.float32)

def prepare_feat_img(img, feats, patch_size=16):
    # Prepare features
    feats = torch.t(feats)

    # Get sizes
    H_patch, W_patch, H_pad, W_pad = img.size[0] // patch_size, img.size[1] // patch_size, img.size[0], img.size[1]
    H_pad_lr, W_pad_lr = H_pad // patch_size, W_pad // patch_size

    # Prepare img
    img = img.resize((H_patch, W_patch), Image.BILINEAR)
    img = img.convert('HSV')
    img = np.array(img)

    # Upscale features to match the resolution
    if (H_patch, W_patch) != (H_pad_lr, W_pad_lr):
        feats = F.interpolate(
            feats.T.reshape(1, -1, H_patch, W_patch), 
            size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
        ).reshape(-1, H_pad_lr * W_pad_lr).T

    return img/255, feats.cpu().detach().numpy()

def get_filter_mask(img, patch_size=16):
    H_patch, W_patch = img.size[0]//patch_size, img.size[1]//patch_size
    img = img.resize((H_patch, W_patch), Image.BILINEAR)
    img_rgb = np.array(img)

    img_lab = img.convert('LAB')
    img_lab = np.array(img_lab)

    img_ycbcr = img.convert('YCbCr')
    img_ycbcr = np.array(img_ycbcr)
    
    img_hsv = img.convert('HSV')
    img_hsv = np.array(img_hsv)

    # Mask in HSV color space
    thresh_s, thresh_v = 200, 200
    lower_thresh_h, upper_thresh_h, lower_thresh_v = 100, 150, 150
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    range_1 = (h >= lower_thresh_h) & (h <= upper_thresh_h).astype(int)
    range_2 = (s >= thresh_s) & (v >= thresh_v).astype(int)
    #range_3 = (s >= thresh_s) & (v >= lower_thresh_v) & (v <= thresh_s) & (h >= lower_thresh_h) & (h <= lower_thresh_v).astype(int)
    
    #mask_hsv = np.logical_and(np.logical_or(range_1, range_3), np.logical_not(range_2)).astype(int)
    mask_hsv = np.logical_not(np.logical_or(range_1, range_2)).astype(int)

    # Mask in RGB color space
    upper_thresh_rgb, lower_thresh_rgb = 200, 100
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    range_4 = (r >= upper_thresh_rgb) & ((g <= lower_thresh_rgb) | (b <= lower_thresh_rgb)).astype(int)
    range_5 = (g >= upper_thresh_rgb) & ((r <= lower_thresh_rgb) | (b <= lower_thresh_rgb)).astype(int)
    range_6 = (b >= upper_thresh_rgb) & ((r <= lower_thresh_rgb) | (g <= lower_thresh_rgb)).astype(int)
    mask_rgb = np.logical_not(np.logical_or(range_4, np.logical_or(range_5, range_6))).astype(int)

    # Mask in YCrCb space
    thresh_5, thresh_6 = [0, 90, 100], [255, 170, 180]
    _, cb, cr = img_ycbcr[:, :, 0], img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]
    range_cb = (cb >= thresh_5[1]) & (cb <= thresh_6[1]).astype(int)
    range_cr = (cr >= thresh_5[2]) & (cr <= thresh_6[2]).astype(int)
    mask_ycbcr = np.logical_and(range_cb, range_cr).astype(int)

    # Mask in LAB space
    thresh_7, thresh_8 = 50, 200
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    range_b2 = ((b >= thresh_7) & (b <= 150)).astype(int)
    range_a = ((a <= thresh_8)).astype(int)
    range_la = ((a >= 200) & (l >= 100) & (l <= 170)).astype(int)
    mask_lab = np.logical_and(np.logical_not(range_la), np.logical_not(range_b2)).astype(int)
    '''
    f, axarr = plt.subplots(4, 2)
    axarr[0][0].imshow(img_rgb)
    axarr[0][1].imshow(mask_rgb)
    axarr[1][0].imshow(img_hsv)
    axarr[1][1].imshow(mask_hsv)
    axarr[2][0].imshow(img_ycbcr)
    axarr[2][1].imshow(mask_ycbcr)
    axarr[3][0].imshow(img_lab)
    axarr[3][1].imshow(mask_lab)
    plt.show()
    '''
    return (mask_hsv & mask_rgb).astype(int)

def highlight_neighbor(W_feat, W_comb):
    W_neighbor = np.ones_like(W_comb)
    h, w = W_comb.shape[0], W_comb.shape[1]
    h_sqroot, w_sqroot = int(h ** 0.5), int(w ** 0.5)
    avg = 0
    local_dict = {}

    for i in range(h):
        if W_feat[i,i] == 0.0:
            continue

        neighbors = [i-h_sqroot-1, i-h_sqroot, i-h_sqroot+1, i-1, i+1, i+h_sqroot-1, i+h_sqroot, i+h_sqroot+1]
        for j in neighbors:
            if j > 0 and j < w:
                avg += W_feat[i, j] / W_feat[i, i]
                local_dict[j] = W_feat[i, j] / W_feat[i, i]
        
        avg = avg / len(local_dict)
        for k, v in local_dict.items():
            if v > avg:
                W_neighbor[i, k] += round(v, 2)

        avg = 0
        local_dict.clear()

    return W_comb * W_neighbor

def compute_affinities(img, img_orig, feats, image_color_lambda=5, patch_size=16):
    # get filtration mask to be applied on image and feature vector
    mask = get_filter_mask(img_orig, patch_size=patch_size)
    mask = mask.reshape(-1)

    # prepare image and features
    image, feats = prepare_feat_img(img, feats, patch_size=patch_size)
    
    # apply binary filer mask on image and feat map
    image_size = image.shape[0]
    image = image.reshape(image_size*image_size, 3) * mask.reshape(-1, 1)
    feats = feats * mask.reshape(-1, 1)
    image = image.reshape(image_size, image_size, 3)

    ### Feature affinities 
    W_feat = (feats @ feats.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()

    ### Color affinities 
    # If we are fusing with color affinites, then load the image and compute
    if image_color_lambda > 0:
        W_lr = knn_affinity(image, n_neighbors=[20, 10, 5], distance_weights=[0.1, 1.0, 2.0])
        W_color = np.array(W_lr.todense().astype(np.float32))
    else:
        # No color affinity
        W_color = 0

    # Combine affinities
    W_comb = W_feat + (W_color * image_color_lambda)

    # Highlight neighboring patches
    #W_comb = highlight_neighbor(W_feat, W_comb)

    return W_comb, mask

def compute_eigs(L, D, K: int = 2):
    # Extract eigenvectors
    try:
        eigenvalues, eigenvectors = eigsh(L, sigma=0, k=K, which='LM', M=D)
    except:
        eigenvalues, eigenvectors = eigsh(L, k=K, which='SM', M=D)
    
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    
    # Sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]

    return eigenvectors

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg  #0
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc


def cut(img_path, backbone, patch_size, fixed_size=480, cpu=False, laplacian="comb", clahe=False, hair=False, thresh=0.1):
    I = Image.open(img_path).convert('RGB')

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    # Data Augmentation
    image = do_aug(I_resize, clahe, hair)
    #image = I_resize

    tensor = ToTensor(image).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]

    # compute combined affinity matrix
    W, mask = compute_affinities(image, I_resize, feat, image_color_lambda=5, patch_size=patch_size)

    # apply filteration mask on affinity matrix
    W = W * np.expand_dims(mask, axis=0)
    W = W * np.expand_dims(mask, axis=1)

    if laplacian == "hodge":
        ############# SIMPLICIAL COMPLEX SECTION
        N = W.shape[0]
        sc = tnx.SimplicialComplex()

        for i in range(N):
            sc.add_node(i)
            for j in range(i + 1, N):
                if W[i, j] > thresh:
                    sc.add_simplex((i,j), weight=W[i, j])
                #for k in range(j + 1, N):
                    #similarity = (W[i, j] +  W[j, k] +  W[k, i]) / 3
                    #if similarity > thresh:
                        #sc.add_simplex((i,j,k), weight=similarity)

        # construct 0-hodge laplacian
        L0 = sc.hodge_laplacian_matrix(0).toarray()
        D = np.array(get_diagonal(L0).todense())
        
        # compute eigenvectors and extract second smallest eigenvector
        eigenvecs = compute_eigs(L0, D, K=2)
        second_eigenvec = eigenvecs[1].numpy()
    else:
        # compute laplacian of the affinity matrix 
        L, D = compute_laplacian(W, lap=laplacian, s=1.0)

        # compute eigenvectors of the laplacian and extract second smallest eigenvector
        eigenvecs = compute_eigs(L, D, K=2)
        second_eigenvec = eigenvecs[1].numpy()

    # get salient area
    bipartition = get_salient_areas(second_eigenvec)

    # check if we should reverse the partition
    dims, scales, init_image_size = [feat_h, feat_w], [patch_size, patch_size], [h,w]
    seed = np.argmax(np.abs(second_eigenvec))
    
    reverse = False
    nc = check_num_fg_corners(bipartition, dims)
    if nc == 4:
        reverse = True

    if reverse:
        # reverse bipartition, eigenvector and get new seed
        second_eigenvec = second_eigenvec * -1
        bipartition = np.logical_not(bipartition)
        seed = np.argmax(second_eigenvec)

    # get pixels corresponding to the seed
    bipartition = bipartition.reshape(dims).astype(float)
    _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
    pseudo_mask = np.zeros(dims)
    pseudo_mask[cc[0],cc[1]] = 1
    pseudo_mask = torch.from_numpy(pseudo_mask)
    if not cpu: pseudo_mask = pseudo_mask.to('cuda')

    # upsample pseudo mask to initial image shape
    bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
    bipartition = bipartition.cpu().numpy()
    bipartition[bipartition <= 0] = 0

    # upsample second smallest eigenvector
    second_eigenvec = second_eigenvec.reshape(dims)
    second_eigenvec = torch.from_numpy(second_eigenvec)
    if not cpu: second_eigenvec = second_eigenvec.to('cuda')
    second_eigenvec = F.interpolate(second_eigenvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()

    return bipartition, second_eigenvec.cpu().numpy(), I_new


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def create_image_info_coco(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.now(datetime.UTC).isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info

def create_annotation_info_coco(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "Pseudo-masks with NCut",
    "url": "https://github.com/tibetakinci",
    "version": "1.0",
    "year": 2024,
    "contributor": "Tibet Akinci",
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

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cut script')
    # model arguments
    parser.add_argument('--vit-model', type=str, default='dino', choices=['dino', 'dinov2', 'sammed', 'sam', 'pretrain'], help='which model')
    parser.add_argument('--vit-arch', type=str, default='base', choices=['base', 'small', 'large', 'huge'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8, 14], help='patch size')
    parser.add_argument('--laplacian', type=str, default='comb', choices=['comb', 'norm', 'rw', 'hodge'], help='how to compute laplacian')
    parser.add_argument('--pretrained-dir', type=str, default='pretrain/dino_s16.pth', help='pretrained backbone directory')

    # data arguments
    parser.add_argument('--imgs-path', type=str, default="../../dataset/ISIC2016/train/imgs", help='path to the images')
    parser.add_argument('--gt-path', type=str, default="../../dataset/ISIC2016/train/orig_gt", help='path to the ground truth')
    parser.add_argument('--dataset-name', type=str, default="isic2016", help='dataset name')
    parser.add_argument('--gt-suffix', type=str, default="", help='suffix for ground truth files')
    parser.add_argument('--clahe', action='store_true', help='use CLAHE data augmentation method')
    parser.add_argument('--hair', action='store_true', help='use hair removal data augmentation method')

    # additional arguments
    parser.add_argument('--out-dir', type=str, default='annotations', help='output directory')
    parser.add_argument('--fixed-size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--eval', action='store_true', help='evaluate masks with gt')

    args = parser.parse_args()

    if args.vit_model == 'dinov2':
        args.patch_size = 14
        args.vit_feat = 'q'
    elif args.vit_model == 'sammed':
        args.vit_arch = 'base'
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

    img_folders = sorted(os.listdir(args.imgs_path))

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)

    image_id, segmentation_id = 1, 1
    output_path = './'
    os.makedirs(output_path, exist_ok=True)
    results = {"dice": [], "miou": [], "acc": []}
    fail_cases = []

    for img_name in tqdm(img_folders):
        # get image path
        img_path = os.path.join(args.imgs_path, img_name)
        # get pseudo-masks for each image using graph cut
        try:
            bipartition, _, I_new = cut(img_path, backbone, args.patch_size, \
                fixed_size=args.fixed_size, cpu=args.cpu, laplacian=args.laplacian, clahe=args.clahe, hair=args.hair)
        except:
            print(f"Skipping {img_name}")
            continue

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

        # LONGEST LENGTH AND PERPENDICULAR WIDTH
        #longest_length, max_width, data = longest_length_and_width(pseudo_mask)
        #plot_contour_length_width(np.array(I), data)
        #draw_legth_width(pseudo_mask)

        # create coco-style image info
        image_info = create_image_info_coco(
            image_id, "{}".format(img_name), (height, width, 3))
        output["images"].append(image_info)         

        # create coco-style annotation info
        annotation_info = create_annotation_info_coco(
            segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
        output["annotations"].append(annotation_info)
        segmentation_id += 1
        image_id += 1
        
        if args.eval:
            pseudo_mask[pseudo_mask == 255] = 1

            if args.dataset_name.lower() == 'ph2':
                gt = Image.open(f"{args.gt_path}/{img_name.split('.')[0]}{args.gt_suffix}.bmp")
            else:
                gt = Image.open(f"{args.gt_path}/{img_name.split('.')[0]}{args.gt_suffix}.png")
            gt = np.asarray(gt)
            gt = np.copy(gt)
            gt[gt == 255] = 1

            mm = metrics(pseudo_mask, gt)
            results["miou"].append(mm["iou"])
            results["acc"].append(mm["acc"])
            results["dice"].append(mm["dice"])

            if mm["dice"] < 0.5:
                fail_cases.append(img_name)

    if args.eval:
        print(f"Dice: {np.mean(results['dice'])}")
        print(f"Acc: {np.mean(results['acc'])}")
        print(f"mIoU: {np.nanmean(results['miou'])}")
        print(len(fail_cases))
        print(fail_cases)
    
    # save annotations
    json_name = '{}/{}_train_fixsize{}_lap{}_vit_{}{}.json'.format(args.out_dir,args.dataset_name, args.fixed_size, args.laplacian, args.vit_arch, args.patch_size)
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    print(f'dumping {json_name}')
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
    