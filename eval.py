import os
import glob
import hydra
import utils
import open3d
import logging
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from s3dis import get_block
from omegaconf import DictConfig
from factory import dataset_factory, model_factory


class2color = {  # for visualization
    'ceiling': [0, 255, 0],
    'floor': [0, 0, 255],
    'wall': [0, 255, 255],
    'beam': [255, 255, 0],
    'column': [255, 0, 255],
    'window': [100, 100, 255],
    'door': [200, 200, 100],
    'table': [170, 120, 200],
    'chair': [255, 0, 0],
    'sofa': [200, 100, 100],
    'bookcase': [10, 200, 100],
    'board': [200, 200, 200],
    'clutter': [50, 50, 50]
}
label2color = np.array(list(class2color.values()))


class EvalWorker:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'eval.log'))
        logging.info('Logs will be saved to %s' % self.cfgs.log.dir)

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for evaluation')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        logging.info('Creating model: %s' % cfgs.model.name)
        self.model = model_factory(self.cfgs).to(device=self.device)

    def run(self):
        if self.cfgs.dataset.name == 'modelnet40':
            self.eval_modelnet40()
        elif self.cfgs.dataset.name == 's3dis':
            self.eval_s3dis()

    @torch.no_grad()
    def eval_modelnet40(self):
        self.model.eval()
        epoch_metrics = dict.fromkeys(self.model.get_metrics().keys(), 0)

        logging.info('Loading test set from %s' % self.cfgs.dataset.root_dir)
        test_dataset = dataset_factory(self.cfgs, split='test')
        test_loader = utils.FastDataLoader(test_dataset, self.cfgs.model.batch_size)

        for i, (inputs, target) in enumerate(test_loader):
            inputs = inputs.to(device=self.device, non_blocking=True)
            target = target.to(device=self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                self.model.forward(inputs, target)

            batch_metrics = self.model.get_metrics()
            epoch_metrics = {k: epoch_metrics[k] + batch_metrics[k] * inputs.size(0) for k in epoch_metrics}
            logging.info('S: [%d/%d] | %s' % (i + 1, len(test_loader), self.model.get_log_string()))

        epoch_metrics = {k: epoch_metrics[k] / len(test_dataset) for k in epoch_metrics}
        logging.info('Statistics on test set: %s' % self.model.get_log_string(epoch_metrics))

    @torch.no_grad()
    def eval_s3dis(self):
        self.model.eval()
        area_pred, area_gt = [], []

        dataset_dir = self.cfgs.dataset.root_dir
        area_dir = os.path.join(dataset_dir, 'Area_%d' % self.cfgs.dataset.test_area)
        room_paths = glob.glob(os.path.join(area_dir, '*.npz'))
        logging.info('Evaluating on S3DIS (Area_%d)...' % self.cfgs.dataset.test_area)

        for idx, room_path in enumerate(room_paths):
            room_name = os.path.basename(room_path).split('.')[0]
            logging.info('[%d/%d] Processing %s' % (idx + 1, len(room_paths), room_name))

            room = np.load(room_path)
            room_xyz, room_rgb, room_gt = room['xyz'], room['rgb'], room['gt']
            room_pred = np.ones_like(room_gt) * -1

            xyz_max = np.amax(room_xyz, axis=0)
            for i in range(int(np.ceil(xyz_max[0])) + 1):
                for j in range(int(np.ceil(xyz_max[1])) + 1):
                    indices, block_data, block_gt = get_block(room_xyz, room_rgb, room_gt, xyz_max, i, j, 'all')
                    if indices.shape[0] == 0:
                        continue

                    inputs = torch.tensor([block_data.astype(np.float32)], device=self.device)
                    target = torch.tensor([block_gt.astype(np.int64)], device=self.device)

                    with torch.cuda.amp.autocast():
                        outputs = self.model.forward(inputs, target)

                    block_acc = self.model.get_accuracy()
                    room_pred[indices] = torch.argmax(outputs, dim=1).cpu()
                    logging.info('Block (%d, %d): n_points = %d, acc = %.2f' % (i, j, len(indices), block_acc))

            # points out of range are regarded as `clutter`
            room_gt[room_pred < 0] = len(label2color) - 1
            room_pred[room_pred < 0] = len(label2color) - 1
            area_pred.append(room_pred)
            area_gt.append(room_gt)

            room_acc = 100.0 * np.equal(room_pred, room_gt).sum() / len(room_gt)
            logging.info('Room %s: n_points = %d, acc = %.2f' % (room_name, len(room_gt), room_acc))

            # save predicted results to *.ply
            pcloud_o3d = open3d.geometry.PointCloud()
            pcloud_o3d.points = open3d.utility.Vector3dVector(room_xyz)
            pcloud_o3d.colors = open3d.utility.Vector3dVector(label2color[room_pred] / 255.0)
            filename = os.path.join(self.cfgs.log.dir, '%s.ply' % room_name)
            open3d.io.write_point_cloud(filename, pcloud_o3d)
            logging.info('Visualized point cloud saved to %s' % filename)

        area_pred, area_gt = np.concatenate(area_pred), np.concatenate(area_gt)
        area_acc = 100.0 * np.equal(area_pred, area_gt).sum() / len(area_gt)
        ious = utils.get_ious(area_pred, area_gt, len(class2color))
        logging.info('Area %d: OA = %.2f, mIoU = %.2f' % (self.cfgs.dataset.test_area, area_acc, np.average(ious)))
        for i in range(len(class2color)):
            logging.info('IoU for class %s:\t%.2f' % (list(class2color.keys())[i], ious[i]))

    def load_ckpt(self):
        logging.info('Loading checkpoint from %s' % self.cfgs.model.resume_path)
        checkpoint = torch.load(self.cfgs.model.resume_path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'])


@hydra.main(config_path='conf', config_name='config')
def main(cfgs: DictConfig):
    # resolve configurations
    if cfgs.dataset.n_workers == 'all':
        cfgs.dataset.n_workers = os.cpu_count()
    if cfgs.log.dir is None:
        cfgs.log.dir = os.getcwd()

    # create workers
    os.chdir(hydra.utils.get_original_cwd())
    if torch.cuda.device_count() == 0:  # CPU
        device = torch.device('cpu')
    else:  # GPU
        device = torch.device('cuda:0')

    worker = EvalWorker(device, cfgs)
    worker.load_ckpt()
    worker.run()


if __name__ == '__main__':
    main()
