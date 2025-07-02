from pp_utils import data_prepare as DP
from os.path import join
import numpy as np
import torch
import os
import torch.utils.data as torch_data
from pp_dataset.data_aug import DataAug


class PCDataset(torch_data.Dataset):
    def __init__(self, mode, config, seq_list=None, info_list=None):
        self.name = 'SemanticKITTI'
        self.total_point = config.total_points

        self.dataset_path = "/dataset/data_odometry_velodyne/dataset"
        self.labels_path = "/dataset/data_odometry_velodyne/data_odometry_labels/dataset"

        self.dataset_path = os.path.join(self.dataset_path, "sequences")
        self.labels_path = os.path.join(self.labels_path, "sequences")
        self.config = config

        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(list(self.label_to_names.keys()))
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.num_valid_classes = self.num_classes - len(self.ignored_labels)

        # for the normalization of the initial data
        # the statistics values are from https://github.com/PRBonn/lidar-bonnetal
        self.data_mean = np.array([10.88, 0.23, -1.04, 0.21, 12.12])  # x,y,z,intensity,range
        self.data_std = np.array([11.47, 6.91, 0.86, 0.16, 12.32])  # x,y,z,intensity,range

        self.mode = mode
        if seq_list is None:
            if mode == 'training':
                seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            elif mode == 'validation':
                seq_list = ['08']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
        if info_list is None:
            info_list = {'xyz': True, 'range': True, 'intensity': True}
        assert isinstance(info_list, dict), "[ERROR] feature list should be dict"
        self.info_list = info_list
        self.data_list = sorted(self.data_list)

        self.da = DataAug(self.config.dug, self.config.rotate_dug,
                          self.config.flip_dug,
                          self.config.scale_dug,
                          self.config.trans_dug,
                          DPA=self.config.drop_dug)

    def get_class_weight(self):
        return DP.get_class_weights(self.name)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        flat_input = self.spatially_regular_gen(item, self.data_list)
        return flat_input

    def spatially_regular_gen(self, item, data_list):
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        pc, intensity, r, labels = self.get_data(pc_path)
        features = []
        normalize_mean = []
        normalize_std = []
        if self.info_list['xyz']:
            features.append(pc)
            normalize_mean.append(self.data_mean[0])
            normalize_std.append(self.data_std[0])
            normalize_mean.append(self.data_mean[1])
            normalize_std.append(self.data_std[1])
            normalize_mean.append(self.data_mean[2])
            normalize_std.append(self.data_std[2])
        if self.info_list['range']:
            features.append(r)
            normalize_mean.append(self.data_mean[-1])
            normalize_std.append(self.data_std[-1])
        if self.info_list['intensity']:
            features.append(intensity)
            normalize_mean.append(self.data_mean[-2])
            normalize_std.append(self.data_std[-2])

        assert len(features) > 0, "[ERROR] No feature be added"
        feature = np.concatenate(features, -1)
        n_mean = np.array(normalize_mean).reshape(-1)
        n_std = np.array(normalize_std).reshape(-1)
        if self.config.f_norm:
            feature = (feature - n_mean) / n_std

        return pc.astype(np.float32), feature.astype(np.float32), \
            labels.astype(np.int32), np.array([cloud_ind], np.int32)

    def get_data(self, file_path):
        seq_id = file_path.split(os.sep)[-3]
        frame_id = file_path.split(os.sep)[-1][:-4]
        data = (np.fromfile(file_path, dtype=np.float32).reshape(-1, 4))

        points = data[:, :3]
        intensity = data[:, 3:]

        ###### label transfer ########
        label_path = join(self.labels_path, seq_id, "labels", frame_id + ".label")
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        ins_labels = labels.copy()
        sem_label = labels & 0xFFFF
        inst_label = labels >> 16
        assert ((sem_label + (inst_label << 16) == labels).all())
        remap_lut = DP.remap_lut
        sem_label = remap_lut[sem_label]
        labels = sem_label.astype(np.int32)
        ################################

        if self.mode == "training":
            points, intensity, labels = self.da.aug(points, intensity, labels)

        r = np.linalg.norm(points, axis=1, keepdims=True)
        cur_num = points.shape[0]
        # shuffle
        random_idx = np.random.permutation(cur_num)

        points = points[random_idx]
        intensity = intensity[random_idx]
        r = r[random_idx]
        labels = labels[random_idx]

        if cur_num > self.total_point:
            spts = torch.from_numpy(points)
            qpts = spts[:1]
            dist = torch.norm(spts - qpts, dim=-1)
            newpts = torch.topk(dist, self.total_point, largest=False, sorted=False).indices.numpy()
            np.random.shuffle(newpts)
            points = points[newpts]
            intensity = intensity[newpts]
            r = r[newpts]
            labels = labels[newpts]

        return points, intensity, r, labels

    @staticmethod
    def collate_fn(batch):
        B = len(batch)
        pc = torch.cat([torch.from_numpy(batch[i][0]) for i in range(B)])
        batch_info = torch.cat([torch.full((len(batch[i][0]),), i, dtype=torch.int32) for i in range(B)], 0)
        feats = torch.cat([torch.from_numpy(batch[i][1]) for i in range(B)])
        labels = torch.cat([torch.from_numpy(batch[i][2]) for i in range(B)])
        cloud_idx = torch.stack([torch.from_numpy(batch[i][3]) for i in range(B)])  # B,1
        batch_data = [pc, batch_info, feats, labels, cloud_idx]
        return batch_data
