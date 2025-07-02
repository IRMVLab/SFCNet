import torch
from pathlib import Path
import os
import numpy as np
import sys

if "linux" in sys.platform and not np.array(['spproj' in lib for lib in torch.ops.loaded_libraries]).any():
    proj_dir = str(Path(__file__).parent.parent.parent.resolve())
    torch.ops.load_library(os.path.join(proj_dir, "lib", "libspproj.so"))

# FLAGS
FLAG_SHIFT = 0b0010
FLAG_EMPTY = 0b0000
FLAG_UP = 0b1000

def fused_frustconv_select_py(qgtensor, qitensor, rgtensor, ritensor, qrange, rrange, idi, batch_size, distance,
                              flag, dilation, kernel_size, spatial_size):
    # only FLAG_SHIFT is available
    M = ritensor[:, 2].max().item() + 1  # [u,v,m,b]
    return torch.ops.spproj.fused_frustconv_select(qgtensor.contiguous(), qitensor.contiguous(),
                                                   rgtensor.contiguous(), ritensor.contiguous(),
                                                   qrange.contiguous(), rrange.contiguous(),
                                                   idi.contiguous(),
                                                   batch_size, distance, flag,
                                                   dilation,
                                                   kernel_size,
                                                   [spatial_size[0], spatial_size[1], M])



def fused_fps_select_py(qgtensor,
                        qitensor,
                        idi,
                        rgtensor,
                        ritensor,
                        random_hw,
                        batch_size,
                        K,
                        distance,
                        flag,
                        dstride,
                        qstride,
                        kernel_size,
                        spatial_size):
    M = ritensor[:, 2].max().item() + 1  # [u,v,m,b]
    out_index = torch.unique(torch.stack(
        [torch.floor(ritensor[:, 0] / dstride[1]).int(), torch.floor(ritensor[:, 1] / dstride[0]).int(),
         ritensor[:, 3]], dim=-1), dim=0)
    return torch.ops.spproj.fused_fps_select(qgtensor.contiguous(), qitensor.contiguous(),
                                             rgtensor.contiguous(), ritensor.contiguous(), idi.contiguous(),
                                             random_hw.contiguous(), out_index.contiguous(),
                                             batch_size, K, distance, flag,
                                             dstride,
                                             qstride,
                                             kernel_size,
                                             [spatial_size[0], spatial_size[1], M])



def spherical_2d_grid(xyz, batch_info, H=64, W=1800, upper_bound=None, lower_bound=None):
    """
    xyz [N,3]
    features [N,C]
    batch_info [N,]
    """
    if upper_bound is None:
        upper_bound = [2.0, 180.]
    if lower_bound is None:
        lower_bound = [-24.8, -180.]

    degree2radian = np.pi / 180
    nLines = H
    AzimuthResolution = (upper_bound[1] - lower_bound[1]) / W
    VerticalViewDown = lower_bound[0]
    VerticalViewUp = upper_bound[0]

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (nLines - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r_square = torch.sum(torch.square(xyz), dim=1)
    r = torch.sqrt(r_square + 1e-10)

    theta = torch.atan2(y, x)

    # alpha
    iCol = (-(lower_bound[1] * degree2radian + theta) / AzimuthResolution).floor().int()

    # beta
    beta = torch.asin(z / r)

    iRow = (H - beta / VerticalResolution - VerticalPixelsOffset).floor().int()

    iRow = torch.clamp(iRow, min=0, max=H - 1)
    iCol = torch.clamp(iCol, min=0, max=W - 1)

    itensor = torch.stack([iCol, iRow, batch_info], dim=-1)  # (u,v,b)

    return_data = [itensor]

    return return_data

def spherical_projection_index_py(xyz, itensor2d, H=64, W=1800):
    """
    xyz [N,3]
    itensor2d [N,3] int [u,v,b]
    return itensor,gtensor """
    return torch.ops.spproj.spherical_projection_index(xyz.contiguous(), itensor2d.contiguous(), [H, W])