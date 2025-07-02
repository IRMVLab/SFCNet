from nuscenes import NuScenes
from nuscenes.utils import splits
from pathlib import Path
import os
from tqdm import tqdm
import pickle as pkl

from pyquaternion import Quaternion
import numpy as np

nuscenes_root = "/dataset/nuScenes2"
pkl_path = "nuscenes_data"


def avaiable_scenes(nusc):
    # only for check if all the files are available
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])

        scene_not_exist = False

        lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
        if not Path(lidar_path).exists():
            scene_not_exist = True

        if scene_not_exist:
            continue
        available_scenes.append(scene)
    return available_scenes


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P


def get_sample_data_calibrate_pose_P(nusc, sample_data, lidar=False):
    sample_data_pose = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sample_data_K = None
    if not lidar:
        sample_data_K = sample_data_pose["camera_intrinsic"]

    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P, sample_data_K


def datagen(nusc, scenes, filename):
    asc = avaiable_scenes(nusc)

    available_scene_names = [s['name'] for s in asc]
    scenes = list(filter(lambda x: x in available_scene_names, scenes))
    scenes = set([asc[available_scene_names.index(s)]['token'] for s in scenes])
    cameras = "CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT".split(' ')
    samples = []
    for sample in tqdm(nusc.sample, total=len(nusc.sample)):
        scene_token = sample['scene_token']
        lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

        if scene_token in scenes:
            lidar_data = nusc.get('sample_data', lidar_token)

            lidar_P = get_sample_data_ego_pose_P(nusc, lidar_data)

            lidar_CP, _ = get_sample_data_calibrate_pose_P(nusc, lidar_data)

            Pcs = []
            imgs = []

            for camera in cameras:
                camera_token = sample["data"][camera]
                camera_data = nusc.get('sample_data', camera_token)

                camera_P = get_sample_data_ego_pose_P(nusc, camera_data)

                camera_CP, camera_K = get_sample_data_calibrate_pose_P(nusc, camera_data)

                camera_CP_inv = np.linalg.inv(camera_CP)
                camera_P_inv = np.linalg.inv(camera_P)

                # T_pc^{cam} @ pc
                Tr = camera_CP_inv @ camera_P_inv @ lidar_P @ lidar_CP

                Pc = camera_K @ Tr[:3]

                imgs.append(camera_data['filename'])
                Pcs.append(Pc)

            lidar_path = lidar_data['filename']
            lidarseg_path = nusc.get('lidarseg', lidar_token)['filename']

            samples.append({
                'pc': lidar_path,
                'images': imgs,
                'label': lidarseg_path,
                'Pc': Pcs
            })

    with open(os.path.join(pkl_path, filename), 'wb') as f:
        pkl.dump(samples, f)


def test_datagen(nusc, scenes, filename):
    asc = avaiable_scenes(nusc)

    available_scene_names = [s['name'] for s in asc]
    scenes = list(filter(lambda x: x in available_scene_names, scenes))
    scenes = set([asc[available_scene_names.index(s)]['token'] for s in scenes])

    samples = []
    for sample in tqdm(nusc.sample, total=len(nusc.sample)):
        scene_token = sample['scene_token']
        lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

        if scene_token in scenes:
            lidar_path = nusc.get('sample_data', lidar_token)['filename']

            samples.append({
                'pc': lidar_path,
                'label': None
            })

    with open(os.path.join(pkl_path, filename), 'wb') as f:
        pkl.dump(samples, f)


if __name__ == '__main__':
    os.makedirs(pkl_path, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=f"{nuscenes_root}/trainval", verbose=True)
    datagen(nusc, splits.train, "train.pkl")
    datagen(nusc, splits.val, "val.pkl")
