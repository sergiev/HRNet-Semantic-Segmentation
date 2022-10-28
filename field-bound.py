import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np

bound_root = 'output/arable_boundaries/baseline_paddle/L1C_T39UVB_A037498_20220827T080236/test_results'
suffix_map = ('arable_boundaries/baseline_paddle','arable_fields/fields_paddle','arable_final')

def postprocess_masks(field_mask, bound_mask):

    bound_mask = bound_mask.astype(np.uint8)

    kernel = np.ones((2, 2), np.uint8)
    bound_mask = cv2.erode(bound_mask, kernel, iterations=1) 

    xor = (field_mask) ^ bound_mask
    final = xor & field_mask
    return final

def vectorize(mask):
    pass

if __name__=='__main__':
    bound_paths = Path(bound_root).glob('*.png')
    final_root = bound_root.replace(suffix_map[0], suffix_map[2])
    Path(final_root).mkdir(parents=True, exist_ok=True)
    for bound_path in tqdm(bound_paths):
        bound_path = str(bound_path)
        bound_mask = cv2.imread(bound_path)
        field_path = bound_path.replace(suffix_map[0], suffix_map[1])
        field_mask = cv2.imread(field_path)
        
        final_mask = postprocess_masks(field_mask, bound_mask)
        final_path = bound_path.replace(suffix_map[0], suffix_map[2])
        cv2.imwrite(final_path, final_mask)