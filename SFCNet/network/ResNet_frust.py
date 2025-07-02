import torch.nn as nn
import torch
import numpy as np
import spconv.pytorch as spconv
from tf_ops.spproj_ops import spherical_projection_index_py, fused_frustconv_select_py, fused_fps_select_py, \
    FLAG_SHIFT, FLAG_EMPTY, spherical_2d_grid
import spconv.pytorch.functional as Fsp
from spconv.tools import CUDAKernelTimer


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        layers = config.block_layers  # [3, 4, 6, 3]
        self.dilation = 1
        self.aux = config.aux
        self.nclasses = config.num_classes
        info_list = config.info_list
        self.init_bn = getattr(config, "init_bn", False)
        input_dim = 0
        if info_list['xyz']:
            input_dim += 3
        if info_list['range']:
            input_dim += 1
        if info_list['intensity']:
            input_dim += 1

        self.inplanes = config.hiden_plane

        self.hiden_plane = config.hiden_plane  # 128

        self.distance = config.conv_dist

        if self.init_bn:
            self.firstbn = nn.BatchNorm1d(input_dim)

        self.conv1 = BasicConv2d(input_dim, self.hiden_plane // 2, kernel_size=config.init_kernel, indice_key='down0',
                                 distance=self.distance[0])
        self.conv2 = BasicConv2d(self.hiden_plane // 2, self.hiden_plane, kernel_size=config.init_kernel,
                                 indice_key='down0',
                                 distance=self.distance[0])
        self.conv3 = BasicConv2d(self.hiden_plane, self.hiden_plane, kernel_size=config.init_kernel, indice_key='down0',
                                 distance=self.distance[0])

        self.layer1 = self._make_layer(self.hiden_plane, layers[0], level=0, indice_key='down0',
                                       distance=self.distance[0], distance_down=self.distance[0])
        self.layer2 = self._make_layer(self.hiden_plane, layers[1], stride=2, level=0, indice_key='down1',
                                       distance=self.distance[0], distance_down=self.distance[1])
        self.layer3 = self._make_layer(self.hiden_plane, layers[2], stride=2, level=1, indice_key='down2',
                                       distance=self.distance[1], distance_down=self.distance[2])
        self.layer4 = self._make_layer(self.hiden_plane, layers[3], stride=2, level=2, indice_key='down3',
                                       distance=self.distance[3], distance_down=self.distance[3])

        self.trans3to0 = FrustConv2D(self.hiden_plane, self.hiden_plane,
                                     kernel_size=self.config.trans_conv_kernel[2], bias=False,
                                     transpose=True, indice_key='3to0', level=3,
                                     distance=self.distance[0])
        self.trans2to0 = FrustConv2D(self.hiden_plane, self.hiden_plane,
                                     kernel_size=self.config.trans_conv_kernel[1], bias=False,
                                     transpose=True, indice_key='2to0', level=2,
                                     distance=self.distance[0])
        self.trans1to0 = FrustConv2D(self.hiden_plane, self.hiden_plane,
                                     kernel_size=self.config.trans_conv_kernel[0], bias=False,
                                     transpose=True, indice_key='1to0', level=1,
                                     distance=self.distance[0])

        self.conv_1 = BasicConv2d(self.config.hiden_plane * 5, self.config.hiden_plane * 2,
                                  kernel_size=config.out_kernel,
                                  indice_key='down0',
                                  distance=self.distance[0])
        self.conv_2 = BasicConv2d(self.config.hiden_plane * 2, self.config.hiden_plane, kernel_size=config.out_kernel,
                                  indice_key='down0',
                                  distance=self.distance[0])
        self.semantic_output = nn.Linear(self.config.hiden_plane, self.nclasses)

        if self.aux:
            self.aux_head1 = nn.Linear(self.config.hiden_plane, self.nclasses)
            self.aux_head2 = nn.Linear(self.config.hiden_plane, self.nclasses)
            self.aux_head3 = nn.Linear(self.config.hiden_plane, self.nclasses)

        if self.config.act == "Hardswish":
            self._convert_act(self, nn.Hardswish())
        elif self.config.act == "SiLU":
            self._convert_act(self, nn.SiLU())

    @staticmethod
    def _convert_act(model, act):
        for child_name, child in model.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(model, child_name, act)
            else:
                Network._convert_act(child, act)

    def _make_layer(self, planes, blocks, stride=1, level=0, indice_key='', distance=-1., distance_down=-1.):
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, level, indice_key,
                                 distance=distance, distance_down=distance_down))
        # self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, indice_key=indice_key,
                                     distance=distance_down, distance_down=distance_down))

        return nn.Sequential(*layers)

    def forward(self, batch_data):
        end_points = {}
        end_points["raw xyz"] = batch_data[0]  # N,3
        end_points["feature"] = batch_data[2]  # N,5
        end_points['labels'] = batch_data[3]  # N,
        end_points["cloud indices"] = batch_data[4]
        end_points["batch_info"] = batch_data[1]

        xyz = batch_data[0]
        feature = batch_data[2]
        label = batch_data[3]
        batch_info = batch_data[1]
        batch_size = batch_data[4].shape[0]

        if self.init_bn:
            feature = self.firstbn(feature)  # for automatic normalization

        init_H_view = self.config.init_H_rm
        init_W_view = self.config.init_W_rm
        itensor_view = spherical_2d_grid(xyz, batch_info, self.config.init_H_rm, self.config.init_W_rm,
                                         self.config.upper_bound2d_rm, self.config.lower_bound2d_rm)[0]  # u,v,b

        itensor_view, gtensor_view = spherical_projection_index_py(xyz, itensor_view,
                                                                   init_H_view, init_W_view)

        x = FrustConvTensor(itensor_view, gtensor_view, feature, [init_H_view, init_W_view],
                            batch_size)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_2, sample_idx2 = self.trans1to0(x_2)
        res_3, sample_idx3 = self.trans2to0(x_3)
        res_4, sample_idx4 = self.trans3to0(x_4)
        res = [x.features, x_1.features, res_2.features, res_3.features, res_4.features]

        out = x.replace_feature(torch.cat(res, dim=1))
        out = self.conv_1(out)
        out = self.conv_2(out)
        out_feat = self.semantic_output(out.features).transpose(0, 1).unsqueeze(0)

        if self.aux:
            res_2 = self.aux_head1(res_2.features).transpose(0, 1).unsqueeze(0)

            res_3 = self.aux_head2(res_3.features).transpose(0, 1).unsqueeze(0)

            res_4 = self.aux_head3(res_4.features).transpose(0, 1).unsqueeze(0)

        if self.aux:
            end_points['logits'] = [res_4, res_3, res_2, out_feat]
            end_points['labels'] = [label, label, label, label]
            end_points['nei_num'] = [x.neighbors_idx_dict["down03x3"] for _ in range(4)]
        else:
            end_points['logits'] = [out_feat]
            end_points['labels'] = [label]
            end_points['nei_num'] = [x.neighbors_idx_dict["down03x3"]]

        return end_points


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, relu=True,
                 indice_key='', level=0, distance=-1):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = FrustConv2D(in_planes, out_planes,
                                kernel_size=kernel_size, stride=stride,
                                dilation=dilation, bias=False, indice_key=indice_key + '3x3'
                                , level=level, distance=distance)
        self.bn = nn.BatchNorm1d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x.replace_feature(self.bn(x.features))
        if self.relu:
            x = x.replace_feature(self.relu(x.features))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, level=0, indice_key='',
                 distance=-1., distance_down=-1.):
        super(BasicBlock, self).__init__()
        first_prefix = '3x3d' if stride > 1 else '3x3'
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = FrustConv2D(inplanes, planes, kernel_size=3, stride=stride,
                                 bias=False, indice_key=indice_key + first_prefix,
                                 level=level, main_sampler=stride > 1, distance=distance)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = FrustConv2D(planes, planes, kernel_size=3,
                                 bias=False, indice_key=indice_key + '3x3', distance=distance_down)

        self.bn2 = nn.BatchNorm1d(planes)
        if stride > 1:
            self.downsample = FrustConv2D(inplanes, planes, kernel_size=1,
                                          stride=stride, level=level, indice_key=indice_key + 'skip')
        else:
            self.register_parameter("downsample", None)
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.stride > 1:
            out, sample_idx = self.conv1(x)
        else:
            out = self.conv1(x)

        out = out.replace_feature(self.relu(self.bn1(out.features)))

        out = self.conv2(out)

        out = out.replace_feature(self.bn2(out.features))
        if self.stride > 1:
            identity = self.downsample(x, sample_idx)
        out = out.replace_feature(self.relu(identity.features + out.features))
        return out


class FrustConvTensor(object):
    def __init__(self, itensor, gtensor, feat, spatial_size, batch_size, neighbors_idx_dict=None):
        self.itensor = itensor
        self.gtensor = gtensor
        self.feat = feat
        self.spatial_size = spatial_size
        self.device = itensor.device
        self.batch_size = batch_size
        self.device = itensor.device
        self.neighbors_idx_dict = neighbors_idx_dict if neighbors_idx_dict is not None else {}

    @property
    def features(self):
        return self.feat

    def __len__(self):
        return self.itensor.shape[0]

    def get_neighbors_idx(self, key):
        if key in self.neighbors_idx_dict.keys():
            return True, self.neighbors_idx_dict[key]
        else:
            return False, None

    def replace_feature(self, feat):
        return FrustConvTensor(self.itensor, self.gtensor,
                               feat, self.spatial_size,
                               self.batch_size,
                               self.neighbors_idx_dict)

    def __repr__(self):
        return f"act_num: {self.gtensor.shape[0]:d} channel: {self.feat.shape[1]:d} " \
               f"spatial_size ({self.spatial_size[0]:d},{self.spatial_size[1]:d}) " \
               f"device: {self.device}"


class FrustConv2D(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, stride=(1, 1),
                 dilation=(1, 1), distance=-1., bias=True, transpose=False,
                 indice_key='', level=0, main_sampler=False):
        super(FrustConv2D, self).__init__()
        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
        self.stride = [stride, stride] if isinstance(stride, int) else stride
        self.d_in = d_in

        self.d_out = d_out
        self.dilation = [dilation, dilation] if isinstance(dilation, int) else dilation
        self.kernel_shape = 1
        self.distance = distance
        self.transpose = transpose
        self.indice_key = indice_key
        self.main_sampler = main_sampler

        self.level = level
        for s in self.kernel_size:
            self.kernel_shape *= s
        self.weight = torch.nn.Parameter(torch.empty(d_out, *self.kernel_size, d_in,
                                                     dtype=torch.float32), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(d_out, dtype=torch.float32), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.conv1x1 = self.kernel_shape == 1
        self.downsample = self.stride[0] > 1 or self.stride[1] > 1
        self._init_param()

    def forward(self, x, idi=None):
        flag, indice_data = x.get_neighbors_idx(self.indice_key)
        if self.transpose and not self.transpose_new:
            flag, indice_data = x.get_neighbors_idx("level{:d}to{:d}_idx".format(0, self.level))
        if idi is None:
            idi = torch.arange(x.gtensor.shape[0], dtype=torch.int32, device=x.device)
            if self.downsample:
                sampled_itensor, sampled_gtensor, sample_idx = fused_fps_select_py(
                    x.gtensor, x.itensor, idi,
                    x.gtensor, x.itensor, torch.arange(1, dtype=torch.int32, device=x.device),
                    x.batch_size, 1, 100,
                    FLAG_EMPTY,
                    self.stride,
                    [1, 1], [1, 1],
                    x.spatial_size)

                idi = sample_idx
                new_ss = [int(np.ceil(x.spatial_size[i] / self.stride[i])) for i in range(2)]
                x.neighbors_idx_dict[self.indice_key + '_dense'] = x.itensor, x.gtensor, x.spatial_size
                x.neighbors_idx_dict[f"level{self.level + 1}_sparse"] = sampled_itensor, sampled_gtensor, new_ss

                # only downsample will save

                if self.level == 0:
                    x.neighbors_idx_dict['level0'] = x.itensor, x.gtensor, x.spatial_size
                    x.neighbors_idx_dict['level0to1'] = sample_idx
                else:
                    x.neighbors_idx_dict['level0to{:d}'.format(self.level + 1)] = \
                        x.neighbors_idx_dict['level0to{:d}'.format(self.level)].gather(0, sample_idx.long())

        if self.transpose:
            level0itensor, level0gtensor, level0spatial_size = x.neighbors_idx_dict["level0"]
        if not self.conv1x1:
            if flag:
                indice_pairs_calc, indice_pair_num = indice_data
            else:
                if self.transpose:
                    idi = torch.arange(level0gtensor.shape[0], dtype=torch.int32, device=x.device)
                    # upsample
                    sparse_itensor, sparse_gtensor, sparse_ss = x.neighbors_idx_dict[f"level{self.level}_sparse"]
                    sh, sw = level0spatial_size[0] // sparse_ss[0], level0spatial_size[1] // sparse_ss[1]
                    sparse_itensor_up = sparse_itensor.clone()  # u,v,b,m
                    sparse_itensor_up[:, 0] *= sw
                    sparse_itensor_up[:, 1] *= sh
                    indice_pairs_calc, indice_pair_num = fused_frustconv_select_py(level0gtensor,
                                                                                   level0itensor,
                                                                                   sparse_gtensor,
                                                                                   sparse_itensor_up,
                                                                                   torch.linalg.norm(
                                                                                       level0gtensor[:, :3],
                                                                                       dim=-1),
                                                                                   torch.linalg.norm(
                                                                                       sparse_gtensor[:, :3],
                                                                                       dim=-1),
                                                                                   idi,
                                                                                   x.batch_size,
                                                                                   self.distance,
                                                                                   FLAG_SHIFT,
                                                                                   self.dilation,
                                                                                   self.kernel_size,
                                                                                   level0spatial_size)

                else:
                    indice_pairs_calc, indice_pair_num = fused_frustconv_select_py(x.gtensor, x.itensor,
                                                                                   x.gtensor, x.itensor,
                                                                                   torch.linalg.norm(x.gtensor[:, :3],
                                                                                                     dim=-1),
                                                                                   torch.linalg.norm(x.gtensor[:, :3],
                                                                                                     dim=-1),
                                                                                   idi,
                                                                                   x.batch_size,
                                                                                   self.distance,
                                                                                   FLAG_SHIFT,
                                                                                   self.dilation,
                                                                                   self.kernel_size,
                                                                                   x.spatial_size)
                x.neighbors_idx_dict[self.indice_key] = indice_pairs_calc, indice_pair_num
                if self.downsample and self.level == 0:
                    x.neighbors_idx_dict['level0to1_idx'] = indice_pairs_calc, indice_pair_num
            bias_infer = self.bias if not self.training else None
            bias_training = self.bias if self.training else None
            algo = spconv.ConvAlgo.Native

            out_features = Fsp.indice_conv(
                x.feat,  # [N,C]
                self.weight,
                indice_pairs_calc,  # [2,K*K,N_out]
                indice_pair_num,  # [K*K]
                indice_pairs_calc.shape[-1],
                algo,
                CUDAKernelTimer(False),
                bias_infer)
            if bias_training is not None:
                out_features += bias_training
        else:
            if self.downsample:
                sample_idx = idi.long().unsqueeze(-1).repeat(1, x.feat.shape[-1])
                sampled_itensor = x.itensor
                sampled_gtensor = x.gtensor
                new_ss = x.spatial_size
                out_features = torch.mm(self.weight.view(self.d_out, self.d_in),
                                        x.feat.gather(0, sample_idx).transpose(0, 1)).transpose(0, 1)
            else:
                # MLP
                out_features = torch.mm(self.weight.view(self.d_out, self.d_in),
                                        x.feat.transpose(0, 1)).transpose(0, 1)
            if self.bias is not None:
                out_features = out_features + self.bias

        if self.downsample:
            out_x = FrustConvTensor(sampled_itensor,
                                    sampled_gtensor,
                                    out_features,
                                    new_ss,
                                    x.batch_size,
                                    x.neighbors_idx_dict)
        else:
            if self.transpose:
                out_x = FrustConvTensor(level0itensor, level0gtensor,
                                        out_features, level0spatial_size,
                                        x.batch_size, x.neighbors_idx_dict)
            else:
                out_x = FrustConvTensor(x.itensor, x.gtensor,
                                        out_features, x.spatial_size,
                                        x.batch_size, x.neighbors_idx_dict)
        if self.main_sampler:
            return out_x, sample_idx
        if self.transpose:
            return out_x, idi
        return out_x

    def _init_param(self):
        self._custom_kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out()
            if fan_in != 0:
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def _calculate_fan_in_and_fan_out(self):
        receptive_field_size = 1
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in self.kernel_size:
            receptive_field_size *= s
        fan_in = self.d_in * receptive_field_size
        fan_out = self.d_out * receptive_field_size
        return fan_in, fan_out

    def _custom_kaiming_uniform_(self,
                                 tensor,
                                 a=0,
                                 mode='fan_in',
                                 nonlinearity='leaky_relu'):
        r"""same as torch.init.kaiming_uniform_, with KRSC layout support
        """
        fan = self._calculate_correct_fan(mode)
        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / np.sqrt(fan)
        bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def _calculate_correct_fan(self, mode):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError(
                "Mode {} not supported, please use one of {}".format(
                    mode, valid_modes))

        fan_in, fan_out = self._calculate_fan_in_and_fan_out()
        return fan_in if mode == 'fan_in' else fan_out

    def extra_repr(self):
        return f"FrustConv({self.d_in:d},{self.d_out:d})[{self.indice_key}]:kernel_size({self.kernel_size[0]:d},{self.kernel_size[1]:d})." \
               f"stride:({self.stride[0]:d},{self.stride[1]:d})," \
               f"dilation:({self.dilation[0]:d},{self.dilation[1]:d})," \
               f"distance:{self.distance:.2f}," \
               f"transpose:{self.transpose}"
