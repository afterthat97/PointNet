import os
import glob
import hydra
import utils
import logging
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from s3dis import prepare_input as prepare_s3dis_input
from omegaconf import DictConfig
from factory import dataset_factory, model_factory


class2color = {  # for visualization
    'clutter': [50, 50, 50],
    'ceiling': [0, 255, 0],
    'floor': [0, 0, 255],
    'wall': [0, 255, 255],
    'beam': [255, 255, 0],
    'column': [255, 0, 255],
    'door': [200, 200, 100],
    'window': [100, 100, 255],
    'table': [170, 120, 200],
    'chair': [255, 0, 0],
    'sofa': [200, 100, 100],
    'bookcase': [10, 200, 100],
    'board': [200, 200, 200],
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

        logging.info('Evaluating on S3DIS (Area_%d)...' % self.cfgs.dataset.test_area)
        area_dir = os.path.join(self.cfgs.dataset.root_dir, 'Area_%d' % self.cfgs.dataset.test_area)
        room_names = [r for r in os.listdir(area_dir) if os.path.isdir(os.path.join(area_dir, r))]

        area_pred, area_gt = [], []
        for idx, room_name in enumerate(sorted(room_names)):
            n_points = np.load(os.path.join(area_dir, '%s.npz' % room_name))['n_points']
            room_xyz, room_pred, room_gt = np.zeros([n_points, 3], np.float32), np.zeros([n_points], int), np.zeros([n_points], int)

            for block_path in glob.glob(os.path.join(area_dir, room_name, 'block_zero_*.npz')):
                data = np.load(block_path)
                block_xyz, block_rgb, block_gt = data['block_xyz'], data['block_rgb'], data['block_gt']
                block_size, room_xyz_max, room2block_indices = data['block_size'], data['room_xyz_max'], data['indices']

                xcenter, ycenter = np.amin(block_xyz, axis=0)[:2] + block_size / 2
                block_data = prepare_s3dis_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max)

                block2split_indices = utils.argsplit(len(block_data), self.cfgs.dataset.n_points)
                for i in range(0, len(block2split_indices), self.cfgs.model.batch_size):
                    block2batch_indices = block2split_indices[i:i+self.cfgs.model.batch_size]
                    inputs = torch.tensor(block_data[block2batch_indices].transpose(0, 2, 1), device=self.device)
                    batch_pred = torch.argmax(self.model.forward(inputs), dim=1).cpu().numpy()

                    room2batch_indices = room2block_indices[block2batch_indices.reshape([-1])]
                    room_xyz[room2batch_indices] = block_xyz[block2batch_indices.reshape([-1])]
                    room_pred[room2batch_indices] = batch_pred.reshape([-1])
                    room_gt[room2batch_indices] = block_gt[block2batch_indices.reshape([-1])]

            room_acc = 100.0 * np.equal(room_pred, room_gt).sum() / len(room_gt)
            area_pred, area_gt = np.hstack([area_pred, room_pred]), np.hstack([area_gt, room_gt])
            logging.info('[%d/%d] %s: acc = %.2f%%' % (idx + 1, len(room_names), room_name, room_acc))

            if 'visualize' in self.cfgs:
                filepath = os.path.join(self.cfgs.log.dir, '%s.ply' % room_name)
                utils.save_ply(filepath, room_xyz, label2color[room_pred] / 255.0)
                logging.info('Visualized result is saved to %s' % filepath)

        area_acc = 100.0 * np.equal(area_pred, area_gt).sum() / len(area_gt)
        ious = utils.get_ious(area_pred, area_gt, len(class2color))
        logging.info('Area %d: OA = %.2f, mIoU = %.2f' % (self.cfgs.dataset.test_area, area_acc, np.average(ious)))
        for i in range(len(class2color)):
            logging.info('IoU for class %s:\t%.2f' % (list(class2color.keys())[i], ious[i]))

    def load_ckpt(self):
        assert self.cfgs.model.resume_path is not None
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
