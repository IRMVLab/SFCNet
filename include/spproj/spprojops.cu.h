// Copyright 2024 IRMVLab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// This head file include the function compiled for cuda
//

#ifndef SPARSECONV_SPPROJOPS_H
#define SPARSECONV_SPPROJOPS_H
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>
#include <cuhash/hash_table.cuh>
#include <tensorview/helper_kernel.cu.h>
#include <tensorview/tensorview.h>
#include <spproj/spprojops.h>


namespace spproj {

    /** for frustum farthest point sampling F2PS **/
    __global__ void stride_based_sampling_fps(tv::TensorView<float> gtensor_new, // N,4
                                              tv::TensorView<int> itensor_new, //N,4
                                              tv::TensorView<const int> itensor,
                                              tv::TensorView<const float> gtensor,
                                              tv::TensorView<int> frust_index,
                                              tv::TensorView<float> frust_dist,
                                              tv::TensorView<const int> out_index, // M,3
                                              tv::TensorView<int> sample_mask,
                                              tv::SimpleVector<int, 2> stride,
                                              tv::SimpleVector<int, 3> ss, // [H,W,M]
                                              unsigned table_size, const cuhash::Entry *table,
                                              cuhash::Functions<4> constants,
                                              uint2 stash_constants, unsigned stash_count
    );

    /** for spherical frustum sparse convolution SFC **/
    __global__ void frust_conv_kernel(tv::TensorView<const float> qgtensor,
                                      tv::TensorView<const int> qitensor,
                                      tv::TensorView<const float> qrange,
                                      tv::TensorView<const float> rgtensor,
                                      tv::TensorView<const float> rrange,
                                      tv::TensorView<const int> idx,
                                      tv::TensorView<int> kernel_idx, // (2,K*K,N) // initial -1,-1
                                      tv::TensorView<int> kernel_number, // (K*K,)
                                      int64_t flag,
                                      float distance,
                                      tv::SimpleVector<int, 2> kernel_size,
                                      tv::SimpleVector<int, 2> dilation,
                                      tv::SimpleVector<int, 3> rss, // include (H,W,KM)
                                      unsigned table_size, const cuhash::Entry *table,
                                      cuhash::Functions<4> constants,
                                      uint2 stash_constants, unsigned stash_count);

}

#endif //SPARSECONV_SPPROJOPS_H
