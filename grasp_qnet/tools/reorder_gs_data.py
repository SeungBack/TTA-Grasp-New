import h5py
import open3d as o3d
import glob
import shutil
import os
from tqdm import tqdm

start_idx = 382410

input_path = '/aihub/OccludedObjectDataset/others/graspnet_1billion/gs_dataset/realsense/train/'
input_path = '/SSDg/shback/graspnet_1billion/gs_dataset/realsense/train/'
# read h5 file
input_paths = sorted(glob.glob(input_path + '*'))

for input_path in tqdm(input_paths):

    try:
        f = h5py.File(input_path, 'r')
        # read data
    except Exception as e:
        print(e)
        os.remove(input_path)
        continue

    # rename
    input_idx = int(input_path.split('/')[-1].split('.')[0])
    if input_idx >= 100000:
        break
    output_idx = start_idx
    shutil.move(input_path, input_path.replace('{:06d}'.format(input_idx), '{:06d}'.format(output_idx)))
    print('Renamed', input_path, 'to', input_path.replace('{:06d}'.format(input_idx), '{:06d}'.format(output_idx)))
    start_idx += 1

