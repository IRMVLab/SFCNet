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

#ifndef SPARSECONV_SPPROJOPS_H
#define SPARSECONV_SPPROJOPS_H

#include <cuhash/hash_table.h>
#include <torch/script.h>
#include <torch_utils.h>
#include <utility/timer.h>
#include <tensorview/tensorview.h>


namespace spproj {
#define KM 20 // The maximum number inside a same 2D position
#define KS 40

    /** for spherical frustum construction **/
    void spherical_assign(const tv::GPU &d,
                         tv::TensorView<float> xi,
                         tv::TensorView<int> im,
                         tv::TensorView<const int> sorted,
                         tv::TensorView<const int> sort_idx);

    /** for spherical frustum construction **/
    std::vector <torch::Tensor> spherical_projection_index(torch::Tensor xyz,
                                                           torch::Tensor itensor2d,
                                                           std::vector <int64_t> spatial_size
    );

    /** for frustum farthest point sampling F2PS **/
    std::vector <torch::Tensor> rulebook_query_frustfps(const tv::GPU &d,
                                                        torch::Tensor &qitensor,
                                                        torch::Tensor &qgtensor,
                                                        torch::Tensor &idx,
                                                        torch::Tensor &ritensor,
                                                        torch::Tensor &rgtensor,
                                                        tv::TensorView<const int> out_index,
                                                        tv::TensorView<const int> random_hw,
                                                        tv::SimpleVector<int, 2> dstride,
                                                        tv::SimpleVector<int, 2> qstride,
                                                        tv::SimpleVector<int, 2> kernel_size,
                                                        tv::SimpleVector<int, 3> ss, // for rss
                                                        float distance,
                                                        int64_t flag,
                                                        int64_t neighbor_number
    );

    /** for frustum farthest point sampling F2PS **/
    std::vector <torch::Tensor> fused_frustfps_select(
            torch::Tensor qgtensor, torch::Tensor qitensor,
            torch::Tensor rgtensor, torch::Tensor ritensor,
            torch::Tensor idx,
            torch::Tensor random_hw,
            torch::Tensor out_index,
            int64_t batch_size,
            int64_t neighbor_number,
            double distance,
            int64_t flag,
            std::vector <int64_t> dstride, // (u,v,m)
            std::vector <int64_t> qstride,
            std::vector <int64_t> kernel_size,
            std::vector <int64_t> spatial_size);

    /** for spherical frustum sparse convolution SFC **/
    std::vector <torch::Tensor> rulebook_query_frustconv(const tv::GPU &d,
                                                         torch::Tensor &qitensor,
                                                         torch::Tensor &qrange,
                                                         torch::Tensor &qgtensor,
                                                         torch::Tensor &idx,
                                                         torch::Tensor &ritensor,
                                                         torch::Tensor &rgtensor,
                                                         torch::Tensor &rrange,
                                                         tv::SimpleVector<int, 2> kernel_size,
                                                         tv::SimpleVector<int, 3> rss,
                                                         tv::SimpleVector<int, 2> dilation,
                                                         float distance, // negative mean no threshold
                                                         int64_t flag
    );

    /** for spherical frustum sparse convolution SFC **/
    std::vector <torch::Tensor> fused_frustconv_select(
            torch::Tensor qgtensor, torch::Tensor qitensor,
            torch::Tensor rgtensor, torch::Tensor ritensor,
            torch::Tensor qrange, torch::Tensor rrange,
            torch::Tensor idx,
            int64_t batch_size,
            double distance,
            int64_t flag,
            std::vector <int64_t> dilation,
            std::vector <int64_t> kernel_size,
            std::vector <int64_t> spatial_size);

}


#endif //SPARSECONV_SPPROJOPS_H
