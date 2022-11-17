from pathlib import Path

import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
import numpy as np
import rasterio as rio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
import cv2

from yacs.config import CfgNode

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from hrnet_ac import get_seg_model

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
    a = [list(folder.rglob(f"{ch}.{ext}")) for ext in extensions for ch in channels]
    return [x[0] for x in a if len(x)]


def read_band_file(file_path) -> np.array:
    return rio.open(file_path).read(1)


def read_bands_from_folder(folder, channels, extensions: List) -> np.array:
    files = parse_dir(folder, channels, extensions)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(read_band_file, files)

    combined = np.dstack(results)
    return combined

def load_model_hrnet(cfg, state_dict, device):
    model = get_seg_model(cfg)
    model.load_state_dict(state_dict)
    model = model.to(device=device)
    return model

# set up constants
CHANNELS = ['BLU', 'GRN', 'RED']
EXT = ['jp2']
if __name__=='__main__':
    padding = 0
    tile_size = 4096
    tile_step = tile_size
    # data_path = Path("/mnt/localssd/semyon/tif_test_15112022")
    data_path = Path("test_geo/L1C_T37UDT_A036526_20220620T084448")
    # data_path = Path("test_geo/L1C_T39UVB_A037498_20220827T080236")

    both_model_path = '9e17e7ac_s2_summer_arable_boundary_011122.pth'

    device = torch.device('cuda')
    ckpt = torch.load(both_model_path, map_location={'cuda': 'cpu'})
    model_field = load_model_hrnet(ckpt["field_cfg"], ckpt["field_state_dict"], device)
    model_bound = load_model_hrnet(ckpt["bound_cfg"], ckpt["bound_state_dict"], device)

    model_bound.cuda()
    model_bound.eval()
    model_field.cuda()
    model_field.eval()

    # read an image as a numpy array
    large_img = read_bands_from_folder(data_path, CHANNELS, EXT)
    if padding > 0:
        large_img = np.pad(large_img, ((padding,padding), (padding,padding), (0,0)))
    # np.pad(qb512[:,:,None],((2,2),(2,2),(0,0))).shape
    # create image slicer
    tiler = ImageSlicer(large_img.shape, tile_size=tile_size, tile_step=tile_step)
    # get coordinates of an each slice
    crops = tiler.crops
    # create tile merger with channels=1(as it's a binary segmentation)
    merger = TileMerger(tiler.target_shape, 1, tiler.weight)

    # list to store predictions
    results = []

    with torch.no_grad():
        # get all tiles from a large raster
        tiles = [tile.astype(np.float32) / 10000.0 for tile in tiler.split(large_img)]
        
        for tile in tqdm(tiles):
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
            
            # substract the bounds
            final = res_field & ~res_bound
            # store the result
            results.append(final)

    # add all predictions to the merger
    merger.integrate_batch(torch.tensor(results), crops)
    # merge slices
    whole_prediction = merger.merge()
    # convert to numpy and move channels dim 
    whole_prediction = np.moveaxis(whole_prediction.detach().cpu().numpy(), 0, -1)
    # crop the mask(as before division the padding is applied)
    final_prediction = tiler.crop_to_orignal_size(whole_prediction).squeeze()
    final_prediction = final_prediction.astype(np.uint8)
    if padding > 0: # unpad
        final_prediction = final_prediction[padding:-padding, padding:-padding]
        
    # get the metadata
    meta = rio.open(data_path / f"{CHANNELS[0]}.{EXT[0]}").meta
    meta["count"] = 1
    meta["dtype"] = "uint8"
    meta["nodata"] = None
    
    # set up output path
    out_path = Path(f"output/arable_imitate/{data_path.name}_{tile_size}.tif")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # save the prediction raster
    with rio.open(out_path, "w", **meta) as f:    
        f.write(final_prediction, 1)

    print(np.unique(final_prediction))
    cv2.imwrite(str(out_path).replace('.tif', '_255.png'), final_prediction.astype(np.uint8) * 255)