import numpy as np
import os, yaml
from os.path import join, dirname, abspath

BASE_DIR = dirname(abspath(__file__))
data_config = join(BASE_DIR, "semantic-kitti.yaml")
DATA = yaml.safe_load(open(data_config, "r"))
remap_dict = DATA['learning_map']
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())


def get_file_list(dataset_path, seq_list):
    data_list = []
    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')
        data_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

    data_list = np.concatenate(data_list, axis=0)
    return data_list


def get_class_weights(dataset_name):
    """
    get the class weight w_c of cross entropy as in Section 3.4
    """
    # pre-calculate the number of points in each category
    num_per_class = []
    if dataset_name is 'SemanticKITTI':
        num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                    240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                    9833174, 129609852, 4506626, 1168181])
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)
    return np.expand_dims(ce_label_weight, axis=0)
