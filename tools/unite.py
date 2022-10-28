from pathlib import Path

import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
import numpy as np
import rasterio as rio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import matplotlib.pyplot as plt
import cv2
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
from pathlib import Path


import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

import _init_paths
import models
import datasets
from config import config
from config import update_config

def parse_args(args):
    update_config(config, args)

    return args

def load_model(cfg_path, weights_path, img_size):
    args = argparse.Namespace(cfg=cfg_path, 
        opts=['TEST.MODEL_FILE', weights_path, 
            'TEST.IMAGE_SIZE', (img_size, img_size), 
            'TEST.BASE_SIZE', img_size])
    args = parse_args(args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)
    model_state_file = config.TEST.MODEL_FILE  

    print('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    
    return model

def inference(model, image, tile_size):
    with torch.no_grad():
        pred = model(image)
        
        ipred = []
        for x in pred:
            x = F.interpolate(
                input=x, size=(tile_size, tile_size),
                mode='bilinear', align_corners=True)
            ipred.append(x)
    
        output = ipred[1].cpu().numpy().transpose(0, 2, 3, 1)
        output[output<0.5] = 0
        output[output>=0.5] = 1
        
        
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    return seg_pred


def tensor2array(tensor):
    return tensor.detach().cpu().numpy()

def parse_dir(folder, channels, extensions):
    return [list(folder.rglob(f"{ch}{ext}"))[0] for ext in extensions for ch in channels]


def read_band_file(file_path) -> np.array:
    return rio.open(file_path).read(1)


def read_bands_from_folder(folder, channels, extensions: List) -> np.array:
    files = parse_dir(folder, channels, extensions)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(read_band_file, files)

    combined = np.dstack(results)
    return combined

# create combined masks

# set up constants
CHANNELS = ['RED', 'GRN', 'BLU', 'NIR']
EXT = ['.jp2']
if __name__=='__main__':
    tile_size = 4096
    tile_step = tile_size

    data_path = Path("/home/qazybek/data/processed/test/FolderCreator/x_steps_config/L1C_T37UDT_A036526_20220620T084448")
    # data_path = Path("/home/qazybek/data/processed/test/FolderCreator/x_steps_config/L1C_T39UVB_A037498_20220827T080236")

    device = torch.device("cuda:0")
    # read an image as a numpy array
    large_img = read_bands_from_folder(data_path, CHANNELS, EXT)

    # create image slicer
    tiler = ImageSlicer(large_img.shape, tile_size=tile_size, tile_step=tile_step)
    # get coordinates of an each slice
    crops = tiler.crops
    # create tile merger with channels=1(as it's a binary segmentation)
    merger = TileMerger(tiler.target_shape, 1, tiler.weight)

    # list to store predictions
    results = []


    bound_cfg = 'experiments/arable_boundaries/baseline_paddle.yaml'
    field_cfg = 'experiments/arable_fields/fields_paddle.yaml'

    bound_weights = 'output/arable_boundaries/baseline_paddle/best.pth'
    field_weights = 'output/arable_fields/fields_paddle/best.pth'

    model_bound = load_model(bound_cfg, bound_weights, tile_size)  # model for segmenting boundaries
    model_field = load_model(field_cfg, field_weights, tile_size)  # model for segmenting fields

    model_bound.eval()
    model_field.eval()

    with torch.no_grad():
        # get all tiles from a large raster
        tiles = [tile.astype(np.int32)[:, :, -2::-1] / 10000.0 for tile in tiler.split(large_img)]

        for j, tile in enumerate(tqdm(tiles)):
            
            # pass the batch through models
            img = torch.tensor(tile).to(device).unsqueeze(0)
            img = torch.moveaxis(img, -1, 1).float()
            img = img.cuda()
            res_field = inference(model_field, img, tile_size)
            res_bound = inference(model_bound, img, tile_size)
            if res_bound.shape == res_field.shape and res_bound.ndim == 3:
                if res_bound.shape[0] == 1:
                    res_field = res_field[0]
                    res_bound = res_bound[0]

            # apply erosion(making thinner)
            kernel = np.ones((2, 2), np.uint8)
            res_bound = cv2.erode(res_bound, kernel, iterations=1) 
            # apply xor
            xor = res_field ^ res_bound
            # apply bitwise and
            final = xor & res_field
            # store the result
            results.append(final)

            
    # add all predictions to the merger
    merger.integrate_batch(torch.tensor(results), crops)
    # merge slices
    merged = merger.merge()
    # convert to numpy and move channels dim 
    merged = np.moveaxis(merged.detach().cpu().numpy(), 0, -1)
    # crop the mask(as before division the padding is applied)
    final_pred = tiler.crop_to_orignal_size(merged).squeeze()
    # get the metadata
    meta = rio.open(data_path / "RED.jp2").meta
    meta["count"] = 1

    # set up output path
    out_path = Path(f"/home/ailab/HRNet-Semantic-Segmentation/output/arable_unite_{tile_size}/{data_path.name}/GT.tif")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # save the prediction raster
    # with rio.open(out_path, "w", **meta) as f:    
    #     f.write(final_pred, 1)
    print(np.unique(final_pred))
    cv2.imwrite(str(out_path), final_pred)
    cv2.imwrite(str(out_path).replace('GT.', 'GT_255.'), (final_pred * 255).astype(np.uint8))
    # display it 
    # plt.imshow(rio.open(f"/home/ailab/HRNet-Semantic-Segmentation/output/arable_unite/{data_path.name}/GT.tif").read(1))
