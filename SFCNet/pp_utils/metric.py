import numpy as np
import threading
from sklearn.metrics import confusion_matrix


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg
        self.lock = threading.Lock()

    def add_data(self, end_points, raw=False, fast=True):
        if not raw:
            logits = end_points['valid_logits']
            labels = end_points['valid_labels']
            pred = logits.max(dim=1)[1]
            pred_valid = pred.detach().cpu().numpy()
            labels_valid = labels.detach().cpu().numpy()
        else:
            pred_valid = end_points['pred'].detach().cpu().numpy().reshape(-1)
            labels_valid = end_points['labels'].detach().cpu().numpy().reshape(-1)

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = fast_hist(pred_valid, labels_valid, self.cfg.num_classes) if fast else \
            confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.lock.acquire()
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)
        self.lock.release()

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / \
                      float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


def fast_hist(pred, label, n):
    k = np.logical_and(label >= 0, label < n)
    bin_count = np.bincount(
        n * label[k].astype(np.int_) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def once_fast_iou(end_points, num_classes=19, raw=False, fast=True):
    if not raw:
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()
    else:
        pred_valid = end_points['pred'].detach().cpu().numpy().reshape(-1)
        labels_valid = end_points['labels'].detach().cpu().numpy().reshape(-1)

    val_total_correct = 0
    val_total_seen = 0

    correct = np.sum(pred_valid == labels_valid)
    val_total_correct += correct
    val_total_seen += len(labels_valid)

    conf_matrix = fast_hist(pred_valid, labels_valid, num_classes) if fast else \
        confusion_matrix(labels_valid, pred_valid, np.arange(0, num_classes, 1))

    gt_classes = np.sum(conf_matrix, axis=1)
    positive_classes = np.sum(conf_matrix, axis=0)
    true_positive_classes = np.diagonal(conf_matrix)

    div_num = gt_classes + positive_classes - true_positive_classes
    mask = np.isclose(div_num, 0)
    neg_mask = np.logical_not(mask)

    iou_list = np.empty((num_classes,))
    iou_list[mask] = 0.
    iou_list[neg_mask] = true_positive_classes[neg_mask] / div_num[neg_mask]

    mean_iou = sum(iou_list) / float(num_classes)

    return mean_iou
