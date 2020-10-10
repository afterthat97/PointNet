import os
import glob
import shutil
import logging
import argparse
import numpy as np
from zipfile import ZipFile
from utils import init_logging


parser = argparse.ArgumentParser()
parser.add_argument('--zip_path', required=True, help='Path to Stanford3dDataset_v1.2_Aligned_Version.zip')
parser.add_argument('--tmp_dir', required=False, default='tmp')
parser.add_argument('--out_dir', required=False, default='datasets')
args = parser.parse_args()


class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
class_dict = dict(zip(class_names, range(len(class_names))))


def load_txts(room_dir):
    """
    Load all *.txt files in `dir_path` and concatenate them.
    :param dirpath: string
    :return: pcloud with shape [n_points, 7] [X, Y, Z, R, G, B, class_id]
    """
    xyz_tot, rgb_tot, gt_tot = [], [], []

    for filename in sorted(glob.glob(os.path.join(room_dir, '*.txt'))):
        class_name = os.path.basename(filename).split('_')[0]
        if class_name not in class_names:
            class_name = 'clutter'

        logging.info('Loading %s' % filename)
        pcloud = np.loadtxt(filename)  # [n_points, 6] [X, Y, Z, R, G, B]

        xyz_tot.append(pcloud[:, 0:3].astype(np.float32))
        rgb_tot.append(pcloud[:, 3:6].astype(np.uint8))
        gt_tot.append(np.ones(pcloud.shape[0], np.uint8) * class_dict[class_name])

    xyz = np.concatenate(xyz_tot, 0)
    rgb = np.concatenate(rgb_tot, 0)
    gt = np.concatenate(gt_tot, 0)

    # the points are shifted before save, the most negative point is now at origin
    xyz = np.float32(xyz - np.amin(xyz, axis=0))

    # sort the points along the x-axis for faster range query
    indices = xyz[:, 0].argsort()
    xyz, rgb, gt = xyz[indices], rgb[indices], gt[indices]

    assert xyz.shape[0] == rgb.shape[0] and rgb.shape[0] == gt.shape[0]
    assert xyz.shape[1] == 3 and rgb.shape[1] == 3 and len(gt.shape) == 1
    assert rgb.dtype == np.uint8 and gt.dtype == np.uint8

    return xyz, rgb, gt


def main():
    init_logging()
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    # Unzip Stanford3dDataset_v1.2_Aligned_Version.zip
    logging.info('Unzipping %s' % args.zip_path)
    with ZipFile(args.zip_path, 'r') as zip_file:
        os.makedirs(args.tmp_dir, exist_ok=True)
        zip_file.extractall(path=args.tmp_dir)

    # Convert dataset from `.txt` to `.npz`
    input_dir = os.path.join(args.tmp_dir, 'Stanford3dDataset_v1.2_Aligned_Version')
    output_dir = os.path.join(args.out_dir, 's3dis')
    for area_name in sorted(list(os.walk(input_dir))[0][1]):
        os.makedirs(os.path.join(output_dir, area_name), exist_ok=True)
        area_dir = os.path.join(input_dir, area_name)
        for room_name in sorted(list(os.walk(area_dir))[0][1]):
            room_dir = os.path.join(area_dir, room_name)
            xyz, rgb, gt = load_txts(os.path.join(room_dir, 'Annotations'))
            output_path = os.path.join(output_dir, area_name, '%s.npz' % room_name)
            np.savez(output_path, xyz=xyz, rgb=rgb, gt=gt)
            logging.info('Converted data has been saved to %s' % output_path)

    # Clean
    logging.info('Cleaning...')
    shutil.rmtree(args.tmp_dir)


if __name__ == '__main__':
    main()
