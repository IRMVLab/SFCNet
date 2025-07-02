import numpy as np
import torch
import os
import torch.utils.data as torch_data
from pp_dataset.data_aug import DataAug
import pickle as pkl
import yaml
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file

class PCDataset(torch_data.Dataset):
    def __init__(self, mode, config, seq_list=None, info_list=None):
        self.name = 'NuScenes'
        self.total_point = config.total_points

        with open("pp_utils/nuscenes.yaml", 'r') as f:
            nuscenes_dict = yaml.safe_load(f)

        self.learn_map = np.vectorize(nuscenes_dict["learning_map"].__getitem__)

        self.dataset_path = "/dataset/nuScenes2" # change to your nuScenes path
        self.data_list = None
        if mode == "training":
            self.dataset_path = os.path.join(self.dataset_path, "trainval")
            with open("nuscenes_data/train.pkl", 'rb') as f:
                self.data_list = pkl.load(f)
        elif mode == "validation":
            self.dataset_path = os.path.join(self.dataset_path, "trainval")
            with open("nuscenes_data/val.pkl", 'rb') as f:
                self.data_list = pkl.load(f)
        else:
            raise NotImplementedError

        self.config = config

        self.label_to_names = {0: 'noise',
                               1: 'barrier',
                               2: 'bicycle',
                               3: 'bus',
                               4: 'car',
                               5: 'construction_vehicle',
                               6: 'motorcycle',
                               7: 'pedestrian',
                               8: 'traffic_cone',
                               9: 'trailer',
                               10: 'truck',
                               11: 'driveable_surface',
                               12: 'other_flat',
                               13: 'sidewalk',
                               14: 'terrain',
                               15: 'manmade',
                               16: 'vegetation'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(list(self.label_to_names.keys()))
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.num_valid_classes = self.num_classes - len(self.ignored_labels)

        self.mode = mode

        if info_list is None:
            info_list = {'xyz': True, 'range': True, 'intensity': True}
        assert isinstance(info_list, dict), "[ERROR] feature list should be dict"
        self.info_list = info_list

        self.da = DataAug(self.config.dug, self.config.rotate_dug,
                          self.config.flip_dug,
                          self.config.scale_dug,
                          self.config.trans_dug)

    def get_class_weight(self):
        return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        flat_input = self.spatially_regular_gen(item, self.data_list)
        return flat_input

    def spatially_regular_gen(self, item, data_list):
        cloud_ind = item
        pc_path = data_list[cloud_ind]

        pc, intensity, r, labels,labels_recon = self.get_data(pc_path)
        features = []
        if self.info_list['xyz']:
            features.append(pc)
        if self.info_list['range']:
            features.append(r)
        if self.info_list['intensity']:
            features.append(intensity)


        assert len(features) > 0, "[ERROR] No feature be added"
        feature = np.concatenate(features, -1)

        return pc.astype(np.float32), feature.astype(np.float32), \
               labels.astype(np.int32), np.array([cloud_ind], np.int32),\
                labels_recon.astype(np.int32)

    def get_data(self, file_path):
        pc_path = os.path.join(self.dataset_path,file_path["pc"])
        label_path = os.path.join(self.dataset_path,file_path["label"])
        data = LidarPointCloud.from_file(pc_path).points.T

        points = data[:, :3]
        intensity = data[:, 3:]

        ###### label transfer ########

        labels = load_bin_file(label_path)
        labels = labels.reshape(-1)
        sem_label = self.learn_map(labels)
        labels = sem_label.astype(np.int32)
        ################################

        # close points filtering during training
        mask_filter = np.logical_not((points[:, 0] < 0.8) & (points[:, 0] > -0.8)
                                     & (points[:, 1] < 2.7) & (points[:, 1] > -2.7))
        # mask_filter = (points[:, 0] < 0.1) & (points[:, 0] > -0.1) & (points[:, 1] < 0.1) & (points[:, 1] > -0.4)

        points = points[mask_filter]
        # labels = labels
        intensity = intensity[mask_filter]
        labels_recon = np.where(mask_filter)[0]

        # if self.mode == "training":
        #     points, intensity, labels = self.da.aug(points, intensity, labels)
        r = np.linalg.norm(points, axis=1, keepdims=True)
        cur_num = points.shape[0]

        # shuffle
        random_idx = np.random.permutation(cur_num)

        points = points[random_idx]
        intensity = intensity[random_idx]
        r = r[random_idx]
        # labels = labels[random_idx]
        labels_recon = labels_recon[random_idx]

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

        return points, intensity, r, labels,labels_recon

    @staticmethod
    def collate_fn(batch):
        B = len(batch)
        pc = torch.cat([torch.from_numpy(batch[i][0]) for i in range(B)])
        batch_info = torch.cat([torch.full((len(batch[i][0]),), i, dtype=torch.int32) for i in range(B)], 0)
        feats = torch.cat([torch.from_numpy(batch[i][1]) for i in range(B)])
        labels = torch.cat([torch.from_numpy(batch[i][2]) for i in range(B)])
        cloud_idx = torch.stack([torch.from_numpy(batch[i][3]) for i in range(B)])  # B,1

        # recon
        nums = np.array([len(batch[i][2]) for i in range(B)])
        nums = np.cumsum(nums) - nums
        labels_recon = torch.cat([torch.from_numpy(batch[i][4]+nums[i]) for i in range(B)])

        return pc, batch_info, feats, labels, cloud_idx,labels_recon
