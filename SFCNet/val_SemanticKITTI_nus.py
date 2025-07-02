import os
import logging
import warnings
import argparse
from tqdm import tqdm
import torch.nn.functional as F

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers [default: 8]')
parser.add_argument('--loss', type=str, default="iou", choices=["iou", "ce"])
parser.add_argument('--save', action="store_true")
parser.add_argument('--config', type=str, default="config_frust_nuscenes")
parser.add_argument('--dataset', type=str, default="nuscenes_trainset_spp_val")
parser.add_argument("--network", default="ResNet_frust", type=str, help="the network to val")
FLAGS = parser.parse_args()

GPU = FLAGS.gpu
LOG_DIR = FLAGS.log_dir
BATCH_SIZE = FLAGS.batch_size
NUM_WORKERS = FLAGS.num_workers
LOSS = FLAGS.loss
NETWORK = FLAGS.network
DATASET = FLAGS.dataset
CFG = FLAGS.config

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["OMP_NUM_THREADS"] = '3'

# my module

from pp_utils.metric import compute_acc, IoUCalculator
from network.loss_func_layer import compute_loss, IOUloss
# determistic operation
from pp_utils.determinstic import set_seed
from importlib import import_module
from pp_utils.prof import Timings

# set random seed
set_seed(0)

import torch
import torch.nn as nn
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

        file = open(os.path.join(LOG_DIR, "config_val.yaml"), mode="w", encoding="utf-8")
        yaml.dump(vars(FLAGS), file)
        file.close()

        self.log_dir = LOG_DIR
        log_fname = os.path.join(LOG_DIR, 'log_val.log')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Validation")
        self.profiler = Timings()

        val_dataset = PCDataset('validation', cfg, info_list=cfg.info_list)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True
        )

        # Network & Optimizer
        self.net = Network(cfg)

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

        class_weights = None

        self.criterion = None
        if LOSS == "ce":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            self.loss_func = compute_loss
        elif LOSS == "iou":
            self.criterion = IOUloss(class_weights)
            self.loss_func = compute_loss

        if os.path.isfile(os.path.join(LOG_DIR, "checkpoints", "best.pt")):
            checkpoint = torch.load(os.path.join(LOG_DIR, "checkpoints", "best.pt"))
            try:
                self.net_single.load_state_dict(checkpoint['model_state_dict'])
            except:
                self.net_single.load_state_dict(checkpoint)
        else:
            raise AttributeError

        # self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if FLAGS.save:
            self.save_path = os.path.join(LOG_DIR, "lidarseg", "val")
            os.makedirs(self.save_path, exist_ok=True)

    def validate(self):
        self.logger.info("**********EVAL*************")
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        acc_sum = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.profiler.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                self.profiler.time("nouse")
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
                self.profiler.time("load_data")

                end_points = self.net(batch_data)

                self.profiler.time("inference")

                label_recon = batch_data[-1]
                logits = end_points['logits'][-1]
                labels = end_points["labels"][-1]

                logits_new = logits.transpose(1, 2).reshape(-1, cfg.num_classes).argmax(-1)
                logits_full = torch.full_like(labels, 3).long()  # as car

                logits_full[label_recon.long()] = logits_new.long()
                valid = (labels > 0)

                logits_full = F.one_hot(logits_full.long()).float()

                if FLAGS.save:
                    num = batch_data[-2].squeeze().item()
                    logits_full.argmax(dim=1).cpu().numpy().tofile(os.path.join(self.save_path,
                                                                                "%06d.label" % num))

                end_points["valid_logits"] = logits_full[valid]
                end_points["valid_labels"] = labels[valid] - 1

                acc, end_points = compute_acc(end_points)
                iou_calc.add_data(end_points)

                acc_sum = (acc_sum * batch_idx + acc.item()) / (batch_idx + 1)

                self.profiler.reset()

        mean_iou, iou_list = iou_calc.compute_iou()

        self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)
        self.logger.info(f"Val acc is {acc_sum:.3f}")
        self.logger.info(self.profiler.summary("Prof:\n"))


def main():
    trainer = Trainer()
    trainer.validate()


if __name__ == '__main__':
    main()
