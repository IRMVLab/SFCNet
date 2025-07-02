import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Lovasz_Softmax import Lovasz_softmax


class IOUloss(nn.Module):
    def __init__(self, weight, w_ce=1., w_ls=1.):
        super(IOUloss, self).__init__()
        self.ls = Lovasz_softmax()
        self.w_ce = w_ce
        self.w_ls = w_ls
        self.crossentropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        pi = F.softmax(logits, dim=-1)  # N,C
        # iou loss
        jacc = self.ls(pi, labels)  # scalar
        wce = self.crossentropy(logits, labels)  # scalar
        return self.w_ce * wce + self.w_ls * jacc





def compute_loss(end_points, dataset, criterion, p_loss=False):
    """
    multi-layer cross-entropy loss
    """
    logits_all = end_points['logits']
    labels_all = end_points['labels']
    N_layer = len(logits_all)

    last_loss = 0.  # last layer's loss with the largest resolution
    loss = 0.
    valid_logits, valid_labels = None, None
    # calculate each layer's label
    for i in range(N_layer):
        logits = logits_all[i]
        labels = labels_all[-i - 1]
        logits = logits.transpose(1, 2).reshape(-1, dataset.num_valid_classes)

        labels = labels.reshape(-1)

        # Boolean mask of points that should be ignored
        ignored_bool = (labels == 0)

        for ign_label in dataset.ignored_labels:
            ignored_bool = ignored_bool | (labels == ign_label)
        ####################### mask invalid ##################
        # Collect logits and labels that are not ignored
        valid_idx = ignored_bool == 0
        valid_logits = logits[valid_idx, :]
        valid_labels_init = labels[valid_idx].long()
        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, dataset.num_valid_classes).long().to(logits.device)
        inserted_value = torch.zeros((1,)).long().to(logits.device)
        for ign_label in dataset.ignored_labels:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        # pick the labels in the reducing_list(0:num) which is valid
        valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
        ####################### mask invalid end ##################
        if i == N_layer - 1:
            # last layer
            loss_last = criterion(valid_logits, valid_labels)
            if len(loss_last.shape) > 0:
                loss_last = loss_last.mean()
            loss += loss_last
            last_loss = loss_last.item()
        else:
            loss_mid = criterion(valid_logits, valid_labels)
            if len(loss_mid.shape) > 0:
                loss_mid = loss_mid.mean()
            loss += loss_mid

    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss

    return loss, end_points, last_loss



