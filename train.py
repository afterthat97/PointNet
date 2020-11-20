import os
import glob
import time
import utils
import hydra
import shutil
import logging
import torch
import torch.optim
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from factory import dataset_factory, model_factory, optimizer_factory


class TrainWorker:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'train.log'))

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            if self.n_gpus > 1:
                dist.init_process_group('nccl', 'tcp://localhost:12345', world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
                self.cfgs.dataset.n_workers = int(self.cfgs.dataset.n_workers / self.n_gpus)
            cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        if self.is_main:
            logging.info('Logs will be saved to %s' % self.cfgs.log.dir)
            self.summary_writer = SummaryWriter(self.cfgs.log.dir)
            self.backup_code()
            logging.info('Configurations:\n' + OmegaConf.to_yaml(self.cfgs))
        else:
            logging.root.disabled = True

        # create DataLoader
        logging.info('Loading training set from %s' % self.cfgs.dataset.root_dir)
        self.train_dataset = dataset_factory(self.cfgs, split='train')
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset) if self.n_gpus > 1 else None
        self.train_loader = utils.FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.dataset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler
        )
        logging.info('Loading test set from %s' % self.cfgs.dataset.root_dir)
        self.val_dataset = dataset_factory(self.cfgs, split='test')
        self.val_loader = utils.FastDataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=False,
            num_workers=self.cfgs.dataset.n_workers,
            pin_memory=True
        )

        logging.info('Creating model: %s' % cfgs.model.name)
        self.model = model_factory(self.cfgs).to(device=self.device)
        self.ddp = DDP(self.model, [self.device.index]) if self.n_gpus > 1 else self.model
        self.best_metrics = None

        logging.info('Creating optimizer: %s' % self.cfgs.training.optimizer)
        self.optimizer, self.lr_scheduler = optimizer_factory(self.cfgs, self.model.parameters())
        self.amp_scaler = torch.cuda.amp.GradScaler()

    def run(self):
        while self.curr_epoch <= self.cfgs.training.max_epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)

            self.train_one_epoch()
            self.validate()
            self.lr_scheduler.step()

            curr_step = self.curr_epoch * len(self.train_loader)
            self.save_summary({'learning_rate': self.optimizer.param_groups[0]['lr']}, curr_step)

            if self.curr_epoch % self.cfgs.log.save_ckpt_every_n_epochs == 0:
                self.save_ckpt()

            self.curr_epoch += 1

    def train_one_epoch(self):
        self.ddp.train()

        start_time = time.time()
        for i, (inputs, target) in enumerate(self.train_loader):
            inputs = inputs.to(device=self.device, non_blocking=True)
            target = target.to(device=self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                self.ddp.forward(inputs, target)
                loss = self.model.get_loss()

            self.optimizer.zero_grad()
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()

            timing = time.time() - start_time
            start_time = time.time()

            logging.info('E: [%d/%d] ' % (self.curr_epoch, self.cfgs.training.max_epochs) +
                         'S: [%d/%d] ' % (i + 1, len(self.train_loader)) +
                         '| %s, timing: %.2fs' % (self.model.get_log_string(), timing))

            if i % self.cfgs.log.save_summary_every_n_steps == 0:
                curr_step = (self.curr_epoch - 1) * len(self.train_loader) + i
                self.save_summary(self.model.get_metrics(), curr_step, prefix='train/')

    @torch.no_grad()
    def validate(self):
        self.ddp.eval()
        epoch_metrics = dict.fromkeys(self.model.get_metrics().keys(), 0)

        start_time = time.time()
        for i, (inputs, target) in enumerate(self.val_loader):
            inputs = inputs.to(device=self.device, non_blocking=True)
            target = target.to(device=self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                self.ddp.forward(inputs, target)

            timing = time.time() - start_time
            start_time = time.time()

            logging.info('S: [%d/%d] ' % (i + 1, len(self.val_loader)) +
                         '| %s, timing: %.2fs' % (self.model.get_log_string(), timing))

            batch_metrics = self.model.get_metrics()
            epoch_metrics = {k: epoch_metrics[k] + batch_metrics[k] * inputs.size(0) for k in epoch_metrics}

        epoch_metrics = {k: epoch_metrics[k] / len(self.val_dataset) for k in epoch_metrics}
        logging.info('Statistics on validation set: %s' % self.model.get_log_string(epoch_metrics))
        self.save_summary(epoch_metrics, self.curr_epoch * len(self.train_loader), prefix='val/')

        if self.model.is_better(epoch_metrics, self.best_metrics):
            self.best_metrics = epoch_metrics
            self.save_ckpt('best.pt')

    def backup_code(self):
        if self.is_main and self.cfgs.log.backup_code:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for pattern in ['*.py', 'models/*.py']:
                for file in glob.glob(pattern):
                    src = os.path.join(base_dir, file)
                    dst = os.path.join(self.cfgs.log.dir, 'backup', os.path.dirname(file))
                    logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy2(src, dst)

    def save_summary(self, summary: dict, curr_step, prefix=''):
        if self.is_main and self.cfgs.log.save_summary:
            for name in summary.keys():
                self.summary_writer.add_scalar(prefix + name, summary[name], curr_step)

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            filepath = os.path.join(ckpt_dir, filename or 'epoch-%03d.pt' % self.curr_epoch)
            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'epoch': self.curr_epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }, filepath)

    def load_ckpt(self):
        if self.cfgs.model.resume_path is not None:
            logging.info('Loading checkpoint from %s' % self.cfgs.model.resume_path)
            checkpoint = torch.load(self.cfgs.model.resume_path, self.device)
            self.curr_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


def create_worker(device_id, cfgs):
    device = torch.device('cpu' if device_id is None else 'cuda:%d' % device_id)
    worker = TrainWorker(device, cfgs)
    worker.load_ckpt()
    worker.run()


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
        create_worker(None, cfgs)
    elif torch.cuda.device_count() == 1:  # Single GPU
        create_worker(0, cfgs)
    elif torch.cuda.device_count() > 1:  # Multiple GPUs
        mp.spawn(create_worker, (cfgs,), torch.cuda.device_count())


if __name__ == '__main__':
    main()
