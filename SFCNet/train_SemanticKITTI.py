"""
The script to train the model
"""
import os
import logging
import time
import warnings
import argparse

import numpy as np
from tqdm import tqdm
import shutil
import platform
from datetime import datetime
import pickle as pkl

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--pretrained_path', default=None, help='Model pretrained path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 150]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--val_batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--dataset', type=str, default="semkitti_trainset_spp")
parser.add_argument('--config', type=str, default="config_frust")
parser.add_argument('--loss', type=str, default="iou", choices=["iou", "ce"])
parser.add_argument("--network", default="ResNet_frust", type=str, help="the network to train")
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers [default: 8]')
parser.add_argument('--grad_norm', type=float, default=-1.)
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
FLAGS = parser.parse_args()

GPU = FLAGS.gpu
CHECKPOINT_PATH = FLAGS.checkpoint_path
PRETRAINED_PATH = FLAGS.pretrained_path
LOG_DIR = FLAGS.log_dir
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
VAL_BATCH_SIZE = FLAGS.val_batch_size
NUM_WORKERS = FLAGS.num_workers
NETWORK = FLAGS.network
DATASET = FLAGS.dataset
LOSS = FLAGS.loss
CFG = FLAGS.config
OPTIM = FLAGS.optim
GN = FLAGS.grad_norm

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["OMP_NUM_THREADS"] = '3'

# my module

from pp_utils.metric import compute_acc, IoUCalculator, once_fast_iou
from network.loss_func_layer import compute_loss, IOUloss
# determistic operation
from pp_utils.determinstic import set_seed
from importlib import import_module
from pp_utils.monitor.base import UniWriter
from pp_utils.prof import Timings

# set random seed
set_seed(0)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml


def my_worker_init_fn(worker_id):
    set_seed(worker_id)


config_path = "pp_utils.{}".format(CFG)
cfg_file = import_module(config_path)
cfg = cfg_file.Config

network_path = "network.{0}".format(NETWORK)
network_file = import_module(network_path)
Network = network_file.Network

dataset_path = "pp_dataset.{}".format(DATASET)
DA = import_module(dataset_path)
PCDataset = DA.PCDataset


class Trainer:
    def __init__(self):
        # Init Logging
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

        file = open(os.path.join(LOG_DIR, "config.yaml"), mode="w", encoding="utf-8")
        yaml.dump(vars(FLAGS), file)
        file.close()

        os.makedirs("{0}/arch".format(LOG_DIR), exist_ok=True)

        if CHECKPOINT_PATH is not None and not LATEST:
            if CHECKPOINT_PATH != LOG_DIR:
                shutil.copy("{0}/arch/network.py".format(CHECKPOINT_PATH), "{0}/arch/network.py".format(LOG_DIR))
                shutil.copy("{0}/arch/dataset.py".format(CHECKPOINT_PATH), "{0}/arch/dataset.py".format(LOG_DIR))
                shutil.copy("{0}/arch/config.py".format(CHECKPOINT_PATH), "{0}/arch/config.py".format(LOG_DIR))
        else:
            shutil.copy("network/{0}.py".format(NETWORK), "{0}/arch/network.py".format(LOG_DIR))
            shutil.copy("pp_dataset/{0}.py".format(DATASET), "{0}/arch/dataset.py".format(LOG_DIR))
            shutil.copy("pp_utils/{}.py".format(CFG), "{0}/arch/config.py".format(LOG_DIR))

        self.log_dir = LOG_DIR
        log_fname = os.path.join(LOG_DIR, 'log_train.log')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Trainer")

        os.makedirs(os.path.join(LOG_DIR, "tensorboard"), exist_ok=True)

        self.writer = UniWriter(log_dir=os.path.join(LOG_DIR, "tensorboard"),
                                suffix=datetime.now().strftime("_train_%Y_%m_%d_%H_%M_%S.tensorboard"))

        train_dataset = PCDataset("training", cfg, info_list=cfg.info_list)
        val_dataset = PCDataset('validation', cfg, info_list=cfg.info_list)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=my_worker_init_fn,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True
        )

        # Network & Optimizer
        self.net = Network(cfg)

        # Multiple GPU Training
        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.net_single = self.net
        if int(FLAGS.gpu) < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.gpu = True
                self.n_gpus = 1
                self.net.to(self.device)

        # Loss Function

        class_weights = None
        class_weight_real = train_dataset.get_class_weight()
        if not cfg.noweightce and class_weight_real is not None:
            class_weights = torch.from_numpy(class_weight_real).float().to(self.device).view(-1)

        self.criterion = None
        if LOSS == "ce":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            self.loss_func = compute_loss
        elif LOSS == "iou":
            self.criterion = IOUloss(class_weights, cfg.w_ce, cfg.w_ls)
            self.loss_func = compute_loss

        # Load the Adam optimizer
        self.step_optim = False
        if OPTIM == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        elif OPTIM == "adamw":
            self.optimizer = optim.AdamW(self.net.parameters(), lr=cfg.learning_rate)
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=cfg.learning_rate,
                                                           epochs=MAX_EPOCH,
                                                           steps_per_epoch=
                                                           len(self.train_loader))
            self.step_optim = True
        else:
            raise NotImplementedError

        # Load module
        self.highest_val_iou = 0
        self.start_epoch = 0

        if CHECKPOINT_PATH is not None and os.path.isfile(os.path.join(CHECKPOINT_PATH, "checkpoints", "ckpt.pt")):
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, "checkpoints", "ckpt.pt"))
            try:
                self.net_single.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch']
                self.highest_val_iou = checkpoint['highest_val_iou']
            except:
                self.net_single.load_state_dict(checkpoint)

        if PRETRAINED_PATH is not None and os.path.isfile(os.path.join(PRETRAINED_PATH, "checkpoints", "best.pt")):
            checkpoint = torch.load(
                os.path.join(PRETRAINED_PATH, "checkpoints", "best.pt"))
            self.net_single.load_state_dict(checkpoint)
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')

        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.profiler = Timings()
        cfg.prof = self.profiler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_one_epoch(self):
        loss_sum = 0
        loss_sum_batch = 0
        batch_i = 0
        miou_sum = 0.
        self.net.train()  # set model to training mode
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        for batch_idx, batch_data in enumerate(tqdm_loader):
            if isinstance(batch_data, dict):
                for key in batch_data.keys():
                    if isinstance(batch_data[key], list):
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(self.device)
                    else:
                        batch_data[key] = batch_data[key].to(self.device)
            else:
                for i in range(len(batch_data)):
                    batch_data[i] = batch_data[i].to(self.device)

            # in pytorch 2.0 True is default https://github.com/pytorch/pytorch/pull/92731
            # def zero_grad(self, set_to_none: bool = True):
            self.optimizer.zero_grad(set_to_none=True)
            end_points = self.net(batch_data)
            loss, end_points, last_loss = self.loss_func(end_points, self.train_dataset, self.criterion, cfg.p_loss)

            loss.backward()

            if GN > 0.:
                nn.utils.clip_grad_norm_(self.net.parameters(), GN)

            self.optimizer.step()

            if self.step_optim:
                self.scheduler.step()

            loss_sum = (loss_sum * batch_idx + last_loss) / (batch_idx + 1)
            loss_sum_batch = (loss_sum_batch * batch_i + last_loss) / (batch_i + 1)

            miou = once_fast_iou(end_points, cfg.num_classes)
            miou_sum = (miou_sum * batch_idx + miou) / (batch_idx + 1)

            tqdm_loader.set_postfix(
                {"miou": "%.2f%% (%.2f%%)" % (miou * 100, miou_sum * 100), "loss": "%.3f" % loss.item()})

            batch_i += 1
        self.logger.info(
            f'train main loss:{loss_sum:.3f} train lr: {self.scheduler.get_last_lr()[-1]:.8f} train miou: {miou_sum * 100:.2f}%')
        if not self.step_optim:
            self.scheduler.step()

        return loss_sum

    def train(self):
        for epoch in range(self.start_epoch, MAX_EPOCH):

            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))

            loss_sum = self.train_one_epoch()
            self.writer.add_scalar("Train/Loss", loss_sum, epoch)

            self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))

            mean_iou, loss_sum, acc_sum = self.validate()
            # Save checkpoint
            self.writer.add_scalar("Val/Loss", loss_sum, epoch)
            self.writer.add_scalar("Val/IoU", mean_iou, epoch)
            self.writer.add_scalar("Val/Acc", acc_sum, epoch)

            checkpoint_file = os.path.join(self.ckpt_dir, 'ckpt.pt')
            if mean_iou > self.highest_val_iou:
                # save best model
                self.highest_val_iou = mean_iou
                model_file = os.path.join(self.ckpt_dir, 'best.pt')
                if platform.system() == "Linux":
                    self.save_model(model_file)
            if platform.system() == "Linux":
                self.save_checkpoint(checkpoint_file)

    def validate(self):

        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        loss_sum = 0
        acc_sum = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                if isinstance(batch_data, dict):
                    for key in batch_data.keys():
                        if isinstance(batch_data[key], list):
                            for i in range(len(batch_data[key])):
                                batch_data[key][i] = batch_data[key][i].to(self.device)
                        else:
                            batch_data[key] = batch_data[key].to(self.device)
                else:
                    for i in range(len(batch_data)):
                        batch_data[i] = batch_data[i].to(self.device)
                # Forward pass
                torch.cuda.synchronize()

                end_points = self.net(batch_data)

                loss, end_points, last_loss = self.loss_func(end_points, self.train_dataset, self.criterion, cfg.p_loss)

                acc, end_points = compute_acc(end_points)
                iou_calc.add_data(end_points)

                loss_sum = (loss_sum * batch_idx + last_loss) / (batch_idx + 1)
                acc_sum = (acc_sum * batch_idx + acc.item()) / (batch_idx + 1)

        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)
        self.logger.info(f"test loss is {loss_sum:.3f} test acc is {acc_sum:.3f}")
        return mean_iou, loss_sum, acc_sum

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            "model_state_dict": self.net_single.state_dict(),
            'highest_val_iou': self.highest_val_iou
        }

        torch.save(save_dict, fname)

    def save_model(self, fname):
        save_dict = self.net_single.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
