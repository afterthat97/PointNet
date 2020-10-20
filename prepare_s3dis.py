import os
import glob
import math
import logging
import argparse
import numpy as np
from zipfile import ZipFile
from utils import init_logging


parser = argparse.ArgumentParser()
parser.add_argument('--zip_path', required=True, help='Path to Stanford3dDataset_v1.2_Aligned_Version.zip')
parser.add_argument('--tmp_dir', required=False, default='tmp-s3dis')
parser.add_argument('--out_dir', required=False, default='datasets')
parser.add_argument('--block_size', required=False, type=float, default=1.0)
parser.add_argument('--grid_size', required=False, type=float, default=0.05)
args = parser.parse_args()


class_names = ['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column', 'door',
               'window', 'table', 'chair', 'sofa', 'bookcase', 'board']
class_dict = dict(zip(class_names, range(len(class_names))))


def load_txts(room_dir):
    room_xyz, room_rgb, room_gt = [], [], []

    for filename in sorted(glob.glob(os.path.join(room_dir, '*.txt'))):
        class_name = os.path.basename(filename).split('_')[0]
        if class_name not in class_names:
            logging.warning('Unknown class "%s" from %s' % (class_name, filename))
            class_name = 'clutter'

        logging.info('Loading %s' % filename)
        pcloud = np.loadtxt(filename)  # [n_points, 6] [X, Y, Z, R, G, B]

        room_xyz.append(pcloud[:, 0:3])
        room_rgb.append(pcloud[:, 3:6])
        room_gt.append(np.ones(pcloud.shape[0]) * class_dict[class_name])

    room_xyz = np.concatenate(room_xyz, 0).astype(np.float32)
    room_rgb = np.concatenate(room_rgb, 0).astype(np.uint8)
    room_gt = np.concatenate(room_gt, 0).astype(np.uint8)
    room_xyz -= np.amin(room_xyz, axis=0)

    # sort points along the x-axis for faster range query
    indices = room_xyz[:, 0].argsort()
    room_xyz, room_rgb, room_gt = room_xyz[indices], room_rgb[indices], room_gt[indices]

    return room_xyz, room_rgb, room_gt


def load_cache(area_dir, room_name):
    cache_path = os.path.join(area_dir, '%s.npz' % room_name)
    if os.path.exists(cache_path):
        logging.info('Loading cache from %s' % cache_path)
        room_data = np.load(cache_path)
        room_xyz, room_rgb, room_gt = room_data['xyz'], room_data['rgb'], room_data['gt']
    else:
        room_dir = os.path.join(area_dir, room_name)
        room_xyz, room_rgb, room_gt = load_txts(os.path.join(room_dir, 'Annotations'))
        logging.info('Saving cache to %s' % cache_path)
        np.savez(cache_path, xyz=room_xyz, rgb=room_rgb, gt=room_gt, n_points=len(room_xyz))
    return np.array(room_xyz, np.float32), np.array(room_rgb, np.uint8), np.array(room_gt, np.uint8)


def room2blocks(room_xyz, offset):
    room_xyz_min = np.amin(room_xyz, axis=0) - [offset, offset, 0]
    block_size = np.array([args.block_size, args.block_size, np.inf])  # Don't split over Z axis.

    block_locations, point_block_indices, block_point_counts = np.unique(
        np.floor((room_xyz - room_xyz_min) / block_size).astype(np.int),
        return_inverse=True, return_counts=True, axis=0)
    block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))

    return block_locations, block_point_indices


def merge_small_blocks(block_locations, block_point_indices):
    block_location2idx = dict()
    for block_idx in range(len(block_locations)):
        block_location = (block_locations[block_idx][0], block_locations[block_idx][1])
        block_location2idx[(block_location[0], block_location[1])] = block_idx

    # merge small blocks into one of their big neighbors
    num_points_per_block_threshold = np.average([len(b) for b in block_point_indices]) / 10
    nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    for block_idx in range(len(block_locations)):
        if len(block_point_indices[block_idx]) >= num_points_per_block_threshold:
            continue

        block_location = (block_locations[block_idx][0], block_locations[block_idx][1])
        for x, y in nbr_block_offsets:
            nbr_block_location = (block_location[0] + x, block_location[1] + y)
            if nbr_block_location not in block_location2idx:
                continue

            nbr_block_idx = block_location2idx[nbr_block_location]
            if len(block_point_indices[nbr_block_idx]) < num_points_per_block_threshold:
                continue

            block_point_indices[nbr_block_idx] = np.concatenate(
                [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
            block_point_indices[block_idx] = np.array([], dtype=np.int)
            break

    non_empty_block_indices = [i for i, b in enumerate(block_point_indices) if len(b) > 0]
    block_locations = [block_locations[i] for i in non_empty_block_indices]
    block_point_indices = [block_point_indices[i] for i in non_empty_block_indices]

    return block_locations, block_point_indices


def block2grids(block_xyz):
    block_xyz_min = np.amin(block_xyz, axis=0)

    grid_locations, point_grid_indices, grid_point_counts = np.unique(
        np.floor((block_xyz - block_xyz_min) / args.grid_size).astype(np.int),
        return_inverse=True, return_counts=True, axis=0)
    grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))

    return grid_locations, grid_point_indices


def uniform_sample_block_points(block_point_indices, room_xyz):
    for block_idx in range(len(block_point_indices)):
        point_indices_in_block = block_point_indices[block_idx]
        grid_locations, grid_point_indices = block2grids(room_xyz[point_indices_in_block])
        grid_point_count_avg = int(np.average([len(g) for g in grid_point_indices]))

        point_indices_in_block_repeated = []
        for grid_idx in range(len(grid_locations)):
            point_indices_in_grid = grid_point_indices[grid_idx]
            repeat_num = math.ceil(grid_point_count_avg / len(point_indices_in_grid))
            if repeat_num > 1:
                point_indices_in_grid = np.tile(point_indices_in_grid, repeat_num)
                point_indices_in_grid = point_indices_in_grid[:grid_point_count_avg]
                np.random.shuffle(point_indices_in_grid)
            point_indices_in_block_repeated.extend(list(point_indices_in_block[point_indices_in_grid]))
        block_point_indices[block_idx] = np.array(point_indices_in_block_repeated, np.int32)

    return block_point_indices


def main():
    input_dir = os.path.join(args.tmp_dir, 'Stanford3dDataset_v1.2_Aligned_Version')
    output_dir = os.path.join(args.out_dir, 's3dis')

    if os.path.isdir(input_dir):
        logging.info('Found the temporary files of the last run, skip unzipping...')
    else:
        logging.info('Unzipping %s' % args.zip_path)
        with ZipFile(args.zip_path, 'r') as zip_file:
            os.makedirs(args.tmp_dir, exist_ok=True)
            zip_file.extractall(path=args.tmp_dir)

    for area_name in sorted(list(os.walk(input_dir))[0][1]):
        os.makedirs(os.path.join(output_dir, area_name), exist_ok=True)
        area_dir = os.path.join(input_dir, area_name)
        for room_name in sorted(list(os.walk(area_dir))[0][1]):
            room_xyz, room_rgb, room_gt = load_cache(area_dir, room_name)
            room_xyz_max = np.amax(room_xyz, axis=0)
            for offset_name, offset in [('zero', 0.0), ('half', args.block_size / 2)]:
                block_locations, block_point_indices = room2blocks(room_xyz, offset)
                logging.info('[%s] [%s] - Room is split into %d blocks. (block size: %.2f)' %
                             (room_name, offset_name, len(block_locations), args.block_size))

                n_blocks_before = len(block_locations)
                block_locations, block_point_indices = merge_small_blocks(block_locations, block_point_indices)
                logging.info('[%s] [%s] - %d of %d blocks are merged.' %
                             (room_name, offset_name, n_blocks_before - len(block_locations), n_blocks_before))

                n_points_before = sum([len(b) for b in block_point_indices])
                block_point_indices = uniform_sample_block_points(block_point_indices, room_xyz)
                logging.info('[%s] [%s] - The number of points increases from %d to %d after resampling.' %
                             (room_name, offset_name, n_points_before, sum([len(b) for b in block_point_indices])))

                output_dir = os.path.join(args.out_dir, 's3dis', area_name, room_name)
                os.makedirs(output_dir, exist_ok=True)
                for block_idx in range(len(block_point_indices)):
                    indices = block_point_indices[block_idx]
                    block_xyz, block_rgb, block_gt = room_xyz[indices], room_rgb[indices], room_gt[indices]
                    block_path = os.path.join(output_dir, 'block_%s_%d.npz' % (offset_name, block_idx))
                    np.savez(block_path,
                             block_xyz=block_xyz, block_rgb=block_rgb, block_gt=block_gt,
                             block_size=args.block_size, indices=indices,
                             n_points_in_block=len(block_xyz), n_points_in_room=len(room_xyz),
                             room_xyz_max=room_xyz_max)
                logging.info('[%s] [%s] - %d blocks are saved under %s' %
                             (room_name, offset_name, len(block_point_indices), output_dir))

            room_path = os.path.join(args.out_dir, 's3dis', area_name, '%s.npz' % room_name)
            np.savez(room_path, xyz=room_xyz, rgb=room_rgb, gt=room_gt, n_points=len(room_xyz))
            logging.info('[%s] - Room (original, %d points) is saved to %s' % (room_name, len(room_xyz), room_path))

            room_path = os.path.join(args.out_dir, 's3dis', area_name, '%s_resampled.npz' % room_name)
            indices = np.sort(np.hstack(block_point_indices))
            np.savez(room_path, xyz=room_xyz[indices], rgb=room_rgb[indices], gt=room_gt[indices], n_points=len(indices))
            logging.info('[%s] - Room (resampled, %d points) is saved to %s' % (room_name, len(indices), room_path))


if __name__ == '__main__':
    init_logging()
    main()
    logging.info('All done.')
