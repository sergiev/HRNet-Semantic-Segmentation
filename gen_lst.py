from pathlib import Path
import os

root = 'data/arable_fields/img/test/'
subdirs = ('all', 'L1C_T37UDT_A036526_20220620T084448', 'L1C_T39UVB_A037498_20220827T080236')

for s in subdirs:
    s_lst = s+'.lst'
    rp = Path(root)/s
    paths = [str(x)[len(root)-9:] for x in rp.glob('*.tif')]
    paths_lst = '\n'.join(paths)
    with open(s_lst, 'w') as file:
        file.write(paths_lst)