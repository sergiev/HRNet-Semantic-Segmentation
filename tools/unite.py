from pathlib import Path

import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
import numpy as np
import rasterio as rio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import cv2
import tifffile as tf


import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

import _init_paths
import models
import datasets
from config import config
from config import update_config
from model_io import load_model

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

def parse_dir(folder, channels, extensions):
    return [list(folder.rglob(f"{ch}{ext}"))[0] for ext in extensions for ch in channels]


def read_band_file(file_path) -> np.array:
    return rio.open(file_path).read(1)


def read_bands_from_folder(folder, channels, extensions: List) -> np.array:
    # N-channel image from separate channel files
    band_files = parse_dir(folder, channels, extensions)
    results = []
    for channel, file in zip(channels, band_files):
        ds = rio.open(file)
        result = ds.read(1)
        results.append(result)
        # meta = ds.meta
        # new_metadata = meta.copy()
        # new_metadata.update(
        #     width=512,
        #     height=512
        # )
        # out_path = folder / f'{channel}_512.tif'
        # result_512 = result[:512,:512]
        # # with rio.open(out_path, 'w', **new_metadata) as file:
        # #     file.write(result_512, 1)
        
    combined = np.dstack(results)
    return combined

def imitate_train_bias(img):
    # optimized version of bias imitation
    # debiased = (img - np.average(img)) / np.std(img)
    # return debiased * TRAIN_STD + TRAIN_AVG
    TRAIN_AVG = 0.18669257
    TRAIN_STD = 0.022204865
    coeff = TRAIN_STD / np.std(img)
    return coeff * (img - np.average(img)) + TRAIN_AVG 

# set up constants
CHANNELS = ['BLU', 'GRN', 'RED']
EXT = ['.jp2']
if __name__=='__main__':
    tile_size = 4096
    tile_step = tile_size

    data_path = Path("test_geo/L1C_T37UDT_A036526_20220620T084448")
    # data_path = Path("/mnt/localssd/semyon/tif_test_15112022")

    device = torch.device("cuda:0")
    # read an image as a numpy array
    large_img = read_bands_from_folder(data_path, CHANNELS, EXT)
    large_img[large_img==-9999]=0
    large_img = large_img.astype(np.float32)
    if np.max(large_img) > 10000:
        large_img /= 10000
    

    # create image slicer
    tiler = ImageSlicer(large_img.shape, tile_size=tile_size, tile_step=tile_step)
    # get coordinates of an each slice
    crops = tiler.crops
    # create tile merger with channels=1(as it's a binary segmentation)
    merger = TileMerger(tiler.target_shape, 1, tiler.weight)
    # get all tiles from a large raster
    tiles = tiler.split(large_img)

    # list to store predictions
    results = []


    bound_cfg = 'experiments/arable_boundaries/baseline_paddle.yaml'
    field_cfg = 'experiments/arable_fields/fields_paddle.yaml'

    bound_weights = 'output/arable_boundaries/baseline_paddle/best.pth'
    field_weights = 'output/arable_fields/fields_paddle/best.pth'

    model_bound = load_model(bound_cfg, bound_weights)  # model for segmenting boundaries
    model_field = load_model(field_cfg, field_weights)  # model for segmenting fields

    model_bound.cuda()
    model_bound.eval()
    model_field.cuda()
    model_field.eval()
    with torch.no_grad():
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
            # kernel = np.ones((2, 2), np.uint8)
            # res_bound = cv2.erode(res_bound, kernel, iterations=1) 
            
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

    # set up output path
    out_path = Path(f"output/arable_unite/{data_path.name}_{tile_size}.tif")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # get the metadata
    # meta = rio.open(data_path / f"RED.{EXT[0]}").meta
    # meta["count"] = 1

    # save the prediction raster
    # with rio.open(out_path, "w", **meta) as f:    
    #     f.write(final_pred, 1)
    final_pred = final_pred.astype(np.uint8)

    print(np.unique(final_pred))
    cv2.imwrite(str(out_path), final_pred)
    cv2.imwrite(str(out_path).replace('.tif', '_255.png'), final_pred * 255)

    # cv2.imwrite('ci0_bands_source.tif', large_img[:tile_size, :tile_size, :3])
    # cv2.imwrite('ci0_bands_source_255.png', (large_img[:tile_size, :tile_size, :3] * 0.0255).astype(np.uint8))
    # cv2.imwrite('ci0_gt_binary.tif', final_pred[:tile_size, :tile_size])
    # cv2.imwrite('ci0_gt_255.png', (final_pred[:tile_size, :tile_size] * 255).astype(np.uint8))

    # display it 
    # plt.imshow(rio.open(f"/home/ailab/HRNet-Semantic-Segmentation/output/arable_unite/{data_path.name}/GT.tif").read(1))
