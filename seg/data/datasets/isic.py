import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from pycocotools.coco import COCO

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.coco import load_sem_seg
#from .coco import load_sem_seg
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
__all__ = ["load_json", "load_file", "register_isic_train", "register_isic_test"]


def load_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    '''
    Load training split of ISIC[2016-2020] datasets with image directory and JSON file containing annotations
    JSON annotations computed by MaskCut or TokenCut.
    Similar to load_coco_json() function from coco.py
    '''
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(coco.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    num_without_valid_segmentation = 0
    dataset_dicts = []
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id

            masks = []
            segm = anno.get("segmentation", None)
            if segm:
                mask = coco.annToMask(anno)
            else:
                logger.warning("{} does not have a valid segmentation mask".format(record["file_name"]))
                num_without_valid_segmentation += 1
                continue  # ignore this mask
            masks.append(mask)
        record["sem_seg_bin_masks"] = masks
        dataset_dicts.append(record)
    
    if num_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} annotations without valid segmentation. ".format(
                num_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def load_gt_file(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    '''
    Load test split of ISIC[2016-2020] datasets with image and ground truth directory.
    File extensions predefined. If any else, please change extension parameters.
    Imported load_sem_seg function from detectron2.data.datasets.coco
    '''
    return load_sem_seg(gt_root, image_root)


def register_isic_train(name, metadata, image_root, label_root):
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    assert isinstance(label_root, (str, os.PathLike)), label_root
    # 1. register a function which returns dicts
    if "sup" in name:
        DatasetCatalog.register(name, lambda: load_gt_file(label_root, image_root))
    else:
        DatasetCatalog.register(name, lambda: load_json(label_root, image_root))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_root=image_root, label_root=label_root, evaluator_type="sem_seg", **metadata
    )

def register_isic_test(name, metadata, image_root, label_root):
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    assert isinstance(label_root, (str, os.PathLike)), label_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_gt_file(label_root, image_root))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_root=image_root, label_root=label_root, evaluator_type="sem_seg", **metadata
    )


if __name__ == "__main__":
    imgs = "../dataset/ISIC2016/train/imgs"
    labels = "../dataset/ISIC2016/train/annotations/isic2016_train_fixsize480_tau0.2_N2_small8_0_100.json"
    load_json(labels, imgs)
    
    '''
    json_file = '../../TokenCut/unsupervised_saliency_detection/output/isic2016_train_tau0.2_small16.json'
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(json_file)
    
    anns = coco.imgToAnns[1]

    for ann in anns:
        segm = ann.get("segmentation", None)
        if segm:
            mask = coco.annToMask(ann)
            print(mask.max())
            mask[mask==1] == 255
            plt.imshow(mask)
            plt.show()
    '''