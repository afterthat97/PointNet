import os
import glob
import shutil
import logging
import argparse
import numpy as np
from zipfile import ZipFile
from utils import init_logging


parser = argparse.ArgumentParser()
parser.add_argument('--zip_path', required=True, help='Path to modelnet40_normal_resampled.zip')
parser.add_argument('--tmp_dir', required=False, default='tmp')
parser.add_argument('--out_dir', required=False, default='datasets')
args = parser.parse_args()


def load_shape_names(dataset_dir):
    filepath = os.path.join(dataset_dir, 'modelnet40_shape_names.txt')
    with open(filepath) as f:
        shape_names = f.read().splitlines()
    shape_names = [c.rstrip() for c in shape_names]
    return shape_names


def load_txt(txt_path):
    pcloud = np.loadtxt(txt_path, delimiter=',')
    pcloud = np.reshape(pcloud, [-1, 6]).astype(np.float32)
    xyz, normal = pcloud[:, 0:3], pcloud[:, 3:6]
    return xyz, normal


def main():
    init_logging()
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    # Unzip modelnet40_normal_resampled.zip
    logging.info('Unzipping %s' % args.zip_path)
    with ZipFile(args.zip_path, 'r') as zip_file:
        os.makedirs(args.tmp_dir, exist_ok=True)
        zip_file.extractall(path=args.tmp_dir)

    # Convert dataset from `.txt` to `.npz`
    input_dir = os.path.join(args.tmp_dir, 'modelnet40_normal_resampled')
    output_dir = os.path.join(args.out_dir, 'modelnet40')
    for shape_name in load_shape_names(input_dir):
        os.makedirs(os.path.join(output_dir, shape_name))
        for txt_path in glob.glob(os.path.join(input_dir, shape_name, '*.txt')):
            model_idx = os.path.basename(txt_path).split('.')[0]
            output_path = os.path.join(output_dir, shape_name, '%s.npz' % model_idx)
            logging.info('Converting %s -> %s' % (txt_path, output_path))
            xyz, normal = load_txt(txt_path)
            np.savez(output_path, xyz=xyz, normal=normal)

    # Copy other files
    for txt_path in glob.glob(os.path.join(input_dir, '*.txt')):
        logging.info('Copying %s -> %s' % (txt_path, output_dir))
        shutil.copy2(txt_path, output_dir)

    # Clean
    logging.info('Cleaning...')
    shutil.rmtree(args.tmp_dir)


if __name__ == '__main__':
    main()
