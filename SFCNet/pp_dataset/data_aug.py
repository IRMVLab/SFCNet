import numpy as np
import random
from scipy.spatial.transform import Rotation
from scipy import ndimage, interpolate
import sys
import pickle as pkl
import yaml


# Elastic distortion
def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0) // gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=False, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


class DataAug(object):
    def __init__(self, AG=False, RA=False, FA=False, SA=False, TA=False, trans_std=None,
                 EA=False, elastic_params=None, DPA=False, drop_rate=0.1):
        self.AG = AG
        self.rotate_aug = RA
        self.flip_aug = FA
        self.scale_aug = SA
        self.transform_aug = TA
        self.elastic_aug = EA
        self.drop_aug = DPA
        self.drop_rate = drop_rate
        if trans_std is None:
            self.trans_std = [0.1, 0.1, 0.1]
        else:
            self.trans_std = trans_std
        if elastic_params is None:
            elastic_params = [[0.12, 0.4], [0.8, 3.2]]
        self.elastic_gran, self.elastic_mag = elastic_params[0], elastic_params[1]

    def aug(self, points, reflection, labels):
        """
        Args:
            points: [N,3]
            reflection: [N]
            labels: [N]
        """
        if self.AG:

            if self.rotate_aug:
                rotate_rad = np.random.random() * 2 * np.pi - np.pi
                c, s = np.cos(rotate_rad), np.sin(rotate_rad)
                j = np.matrix([[c, s], [-s, c]])
                points[:, :2] = np.dot(points[:, :2], j)

            if self.flip_aug:
                flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    points[:, 0] = -points[:, 0]
                elif flip_type == 2:
                    points[:, 1] = -points[:, 1]
                elif flip_type == 3:
                    points[:, :2] = -points[:, :2]
            if self.scale_aug:
                noise_scale = np.random.uniform(0.95, 1.05)
                points[:, 0] = noise_scale * points[:, 0]
                points[:, 1] = noise_scale * points[:, 1]
            # convert coordinate into polar coordinates

            if self.transform_aug:
                noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                            np.random.normal(0, self.trans_std[1], 1),
                                            np.random.normal(0, self.trans_std[2], 1)]).T

                points[:, 0:3] += noise_translate
            if self.elastic_aug:
                points[:, 0:3] = elastic(points[:, 0:3], self.elastic_gran[0], self.elastic_mag[0])
                points[:, 0:3] = elastic(points[:, 0:3], self.elastic_gran[1], self.elastic_mag[1])
            if self.drop_aug:
                points_to_drop = np.random.randint(0, len(points) - 1, int(len(points) * self.drop_rate))
                points = np.delete(points, points_to_drop, axis=0)
                reflection = np.delete(reflection, points_to_drop, axis=0)
                labels = np.delete(labels, points_to_drop, axis=0)

        return points, reflection, labels

