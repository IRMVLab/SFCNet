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

// include the cuda functions
#include "spproj/spprojops.h"
#include "spproj/spprojops.cu.h"
#include <cuhash/hash_table.cuh>
#include <tensorview/helper_launch.h>
#include <tensorview/helper_kernel.cu.h>

namespace spproj {
    __global__ void frust_conv_kernel(tv::TensorView<const float> qgtensor,
                                      tv::TensorView<const int> qitensor,
                                      tv::TensorView<const float> qrange,
                                      tv::TensorView<const float> rgtensor,
                                      tv::TensorView<const float> rrange,
                                      tv::TensorView<const int> idx,
                                      tv::TensorView<int> kernel_idx,    // (2,K*K,N) // initial -1,-1
                                      tv::TensorView<int> kernel_number, // (K*K,)
                                      int64_t flag,
                                      float distance,
                                      tv::SimpleVector<int, 2> kernel_size,
                                      tv::SimpleVector<int, 2> dilation,
                                      tv::SimpleVector<int, 3> rss, // include (H,W,KM)
                                      unsigned table_size, const cuhash::Entry *table,
                                      cuhash::Functions<4> constants,
                                      uint2 stash_constants, unsigned stash_count) {
        /** kernel implementation of spherical frustum sparse convolution **/
        int kernelVolume = kernel_size[0] * kernel_size[1];
        int numGather = idx.dim(0);
        int dilation_h = dilation[0];
        int dilation_w = dilation[1];
        int kernel_w = kernel_size[1];
        int half_kernel_h = kernel_size[0] / 2;
        int half_kernel_w = kernel_size[1] / 2;
        int rH = rss[0];
        int rW = rss[1];
        int rM = rss[2];
        for (int ix: tv::KernelLoopX<int>(numGather)) { // loop

            // query point info
            int q_idx = idx(ix);

            float r_c = qrange(q_idx);

            int u_c = qitensor(q_idx, 0);
            int v_c = qitensor(q_idx, 1);
            int b_c = qitensor(q_idx, 3);

            for (int i = 0; i < kernelVolume; ++i) {
                int locationidx = 0;
                int query_u;
                int query_v;
                query_u = u_c + (i % kernel_w - half_kernel_w) * dilation_w;
                query_v = v_c + (half_kernel_h - i / kernel_w) * dilation_h;

                // valid spatial location check
                if (flag & 0x2) // bit boolean 0x2 == 0b0010
                {
                    // only skip the point which is over the boundary along the H axis
                    if ((query_v < 0) ||
                        (query_v >= rH)) //  the region of padding points (not valid)
                    {
                        continue;
                    }
                    // along the width axis, they should be treated as a circle shift
                    if (query_u < 0) {
                        query_u = rW + query_u; // circle shift
                    }
                    if (query_u >= rW) {
                        query_u = query_u - rW; // circle shift
                    }
                } else {
                    if ((query_v < 0) ||
                        (query_v >= rH) || (query_u < 0) ||
                        (query_u >= rW)) //  the region of padding points (not valid)
                    {
                        continue;
                    }
                }
                int index = b_c * rH * rW * rM + query_v * rW * rM + query_u * rM + 0;
                auto val = cuhash::retrieve((unsigned) (index), table_size,
                                            table, constants, stash_constants, stash_count);
                if (val == cuhash::kNotFound) {
                    // invalid loc
                    continue;
                } else {
                    int nidx = (int) val;
                    float r_q = rrange(nidx);
                    float i_q = rgtensor(nidx, 3);
                    float r_diff_min = 1000.;
                    int min_idx = -1;
                    float r_diff = r_q > r_c ? r_q - r_c : r_c - r_q;
                    if (distance < 0. || r_diff < distance) {

                        if (r_diff < r_diff_min) {
                            r_diff_min = r_diff;
                            min_idx = nidx;
                        }
                    }

                    while (i_q > 0.5) {
                        locationidx += 1;
                        index = b_c * rH * rW * rM + query_v * rW * rM + query_u * rM + locationidx;
                        val = cuhash::retrieve((unsigned) (index), table_size,
                                               table, constants, stash_constants, stash_count);
                        nidx = (int) val;
                        r_q = rrange(nidx);
                        i_q = rgtensor(nidx, 3);
                        r_diff = r_q > r_c ? r_q - r_c : r_c - r_q;
                        if (distance < 0. || r_diff < distance) {

                            if (r_diff < r_diff_min) {
                                r_diff_min = r_diff;
                                min_idx = nidx;
                            }
                        }
                    }
                    if (min_idx >= 0) {
                        int nloc = atomicAdd(&kernel_number(i), 1);
                        kernel_idx(1, i, nloc) = ix; // out
                        kernel_idx(0, i, nloc) = min_idx;
                    }
                }
            }
        }
    }

    // define cpp api func and cuda kernel function
    // fill the value == arange
    __global__ void arangeKernel(unsigned *data, int size) {
        for (int ix: tv::KernelLoopX<int>(size)) {
            data[ix] = ix;
        }
    }

    // for the index preparation
    __global__ void arangeIndexKernel(unsigned *data, tv::TensorView<const int> it, int size, int H, int W, int M) {
        for (int ix: tv::KernelLoopX<int>(size)) {
            // (u,v,m,b)
            data[ix] = it(ix, 3) * H * W * M + it(ix, 1) * W * M + it(ix, 0) * M +
                       it(ix, 2);
        }
    }

    __global__ void spherical_assign_kernel(tv::TensorView<float> xi,
                                           tv::TensorView<int> im,
                                           tv::TensorView<const int> sorted,
                                           tv::TensorView<const int> sort_idx) {
        int numAct = sort_idx.dim(0);
        for (int ix: tv::KernelLoopX<int>(numAct)) {
            // where in original
            int idx = sort_idx(ix);
            int v = sorted(ix);
            int less_num = 0;
            int greater_num = 0;
            for (int i = ix - 1; i >= 0 && sorted(i) == v; --i) {
                if (sort_idx(i) < idx)
                    less_num += 1;
                else
                    greater_num += 1;
            }
            for (int i = ix + 1; i < numAct && sorted(i) == v; ++i) {
                if (sort_idx(i) < idx)
                    less_num += 1;
                else
                    greater_num += 1;
            }
            im(idx) = less_num;
            if (greater_num == 0)
                xi(idx, 0) = 0.;
        }
    }

    void spherical_assign(const tv::GPU &d,
                         tv::TensorView<float> xi,
                         tv::TensorView<int> im,
                         tv::TensorView<const int> sorted,
                         tv::TensorView<const int> sort_idx) {
        int numAct = sorted.dim(0);
        spherical_assign_kernel<<<tv::launch::getBlocks(numAct), tv::launch::CUDA_NUM_THREADS, 0, d.getStream()>>>(
                xi, im, sorted, sort_idx);
        TV_CHECK_CUDA_ERR_V2("spherical assign failed");
    }

    __global__ void stride_based_sampling_fps(tv::TensorView<float> gtensor_new, // N,4
                                              tv::TensorView<int> itensor_new,   // N,4
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
                                              uint2 stash_constants, unsigned stash_count) {
        /** kernel implementation of frustum farthest point sampling */
        int numOut = out_index.dim(0);
        int H = ss[0];
        int W = ss[1];
        int M = ss[2];
        int sH = stride[0];
        int sW = stride[1];

        for (int ix: tv::KernelLoopX<int>(numOut)) {
            int u_d = out_index(ix, 0);
            int v_d = out_index(ix, 1);
            int b_d = out_index(ix, 2);
            int n = 0;
            for (int i = 0; i < sH * sW; ++i) {
                int v_up = v_d * sH + i / sW;
                int u_up = u_d * sW + i % sW;
                int v = 0;
                if (v_up >= H || u_up >= W)
                    continue;
                int index = b_d * H * W * M + v_up * W * M + u_up * M + 0;
                auto val = cuhash::retrieve((unsigned) (index), table_size,
                                            table, constants, stash_constants, stash_count);
                if (val == cuhash::kNotFound) {
                    continue;
                } else {
                    frust_index(ix, n) = (int) val;
                    n += 1;
                    float xi = gtensor((int) val, 3);
                    while (xi > 0.5) {
                        v += 1;
                        index = b_d * H * W * M + v_up * W * M + u_up * M + v;
                        val = cuhash::retrieve((unsigned) (index), table_size,
                                               table, constants, stash_constants, stash_count);
                        if (val == cuhash::kNotFound)
                            break;
                        frust_index(ix, n) = (int) val;
                        n += 1;
                        xi = gtensor((int) val, 3);
                    }
                }
            }
            int sample_num = (int) ceil(((float) n) / (sH * sW));
            itensor_new(frust_index(ix, 0), 0) = u_d;
            itensor_new(frust_index(ix, 0), 1) = v_d;
            itensor_new(frust_index(ix, 0), 2) = 0;
            gtensor_new(frust_index(ix, 0), 3) = 0.;
            sample_mask(frust_index(ix, 0)) = 1;
            for (int j = 1; j < sample_num; ++j) {
                int vo = frust_index(ix, j - 1);
                int mid = j;
                float mdist = -1.;
                for (int k = j; k < n; ++k) {
                    int vv = frust_index(ix, k);
                    float dis = (gtensor(vv, 0) - gtensor(vo, 0)) * (gtensor(vv, 0) - gtensor(vo, 0)) +
                                (gtensor(vv, 1) - gtensor(vo, 1)) * (gtensor(vv, 1) - gtensor(vo, 1)) +
                                (gtensor(vv, 2) - gtensor(vo, 2)) * (gtensor(vv, 2) - gtensor(vo, 2));
                    frust_dist(ix, k) = frust_dist(ix, k) > dis ? dis : frust_dist(ix, k);
                    if (frust_dist(ix, k) > mdist) {
                        mdist = frust_dist(ix, k);
                        mid = k;
                    }
                }
                int tmp = frust_index(ix, j);
                float tmp_d = frust_dist(ix, j);
                frust_index(ix, j) = frust_index(ix, mid);
                frust_dist(ix, j) = frust_dist(ix, mid);
                frust_index(ix, mid) = tmp;
                frust_dist(ix, mid) = tmp_d;
                itensor_new(frust_index(ix, j), 0) = u_d;
                itensor_new(frust_index(ix, j), 1) = v_d;
                itensor_new(frust_index(ix, j), 2) = j;
                gtensor_new(frust_index(ix, j), 3) = 0.;
                sample_mask(frust_index(ix, j)) = 1;
                gtensor_new(frust_index(ix, j - 1), 3) = 1.;
            }
        }
    }

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
                                                        int64_t neighbor_number) {
        auto device = ritensor.device();
        /*create the new_itensor */
        /*first do downsampling*/
        int numDense = rgtensor.size(0);
        int numOut = out_index.dim(0);
        auto sample_mask = torch::zeros({numDense},
                                        torch::dtype(torch::kInt32).device(device));

        auto rgtensor_new = rgtensor.clone();
        auto ritensor_new = ritensor.clone();
        auto ritensor_tv = tv::torch2tv<const int>(ritensor);
        auto rgtensor_tv = tv::torch2tv<const float>(rgtensor);

        auto frust_index = torch::zeros({numOut, dstride[0] * dstride[1] * ss[2]},
                                        torch::dtype(torch::kInt32).device(ritensor.device()));
        auto frust_dist = torch::full({numOut, dstride[0] * dstride[1] * ss[2]}, std::numeric_limits<float>::max(),
                                      torch::dtype(torch::kFloat32).device(ritensor.device()));

        // build the dense hash map
        ///////////////////////////////////////
        auto table = cuhash::HashTable();
        float initial_size = 2.;
        table.Initialize(numDense, initial_size, 4);

        // hash pair
        unsigned *d_keys = nullptr;
        unsigned *d_values = nullptr;
        cudaMalloc((void **) &d_values, sizeof(unsigned) * numDense);
        TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");
        // prepare the input hash
        arangeKernel<<<tv::launch::getBlocks(numDense), tv::launch::CUDA_NUM_THREADS, 0,
        d.getStream()>>>(d_values, numDense);
        TV_CHECK_CUDA_ERR_V2("assignKernel failed");

        cudaMalloc((void **) &d_keys, sizeof(unsigned) * numDense);
        TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");

        arangeIndexKernel<<<tv::launch::getBlocks(numDense), tv::launch::CUDA_NUM_THREADS, 0,
        d.getStream()>>>(d_keys, ritensor_tv, numDense, ss[0], ss[1], ss[2]);
        TV_CHECK_CUDA_ERR_V2("assignIndexKernel failed");

        bool res = table.Build(numDense, d_keys, d_values);
        while (!res and initial_size < 4.) {
            initial_size += 0.5;
            table.Initialize(numDense, initial_size, 4);
            res = table.Build(numDense, d_keys, d_values);
        }
        TV_ASSERT_RT_ERR(res, "Table Build failed")

        cudaFree(d_keys);
        cudaFree(d_values);

        auto tableSize = table.get_table_size();
        auto tableData = table.data();
        auto constants = table.get_constants_4();
        auto stash_constants = table.get_stash_constants();
        auto stash_count = table.get_stash_count();

        //////////////////////////////////////
        stride_based_sampling_fps<<<tv::launch::getBlocks(
                numOut),
        tv::launch::CUDA_NUM_THREADS, 0, d.getStream()>>>(
                tv::torch2tv<float>(rgtensor_new), tv::torch2tv<int>(ritensor_new),
                ritensor_tv, rgtensor_tv,
                tv::torch2tv<int>(frust_index), tv::torch2tv<float>(frust_dist), out_index,
                tv::torch2tv<int>(sample_mask),
                dstride,
                ss,
                tableSize, tableData, constants, stash_constants,
                stash_count);

        TV_CHECK_CUDA_ERR_V2("stride based sampling failed");
        auto sample_idx = torch::nonzero(sample_mask);
        auto repeat_sample_idx = sample_idx.repeat({1, 4});
        sample_idx = sample_idx.squeeze().toType(torch::kInt32);
        auto sampled_gtensor = torch::gather(rgtensor_new, 0, repeat_sample_idx);
        auto sampled_itensor = torch::gather(ritensor_new, 0, repeat_sample_idx);

        return {sampled_itensor, sampled_gtensor, sample_idx};

    }

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
                                                         int64_t flag) {
        int numOut = idx.size(0);
        int numDense = ritensor.size(0);
        // build the dense hash map
        ///////////////////////////////////////
        auto table = cuhash::HashTable();
        float initial_size = 2.;
        table.Initialize(numDense, initial_size, 4);

        // hash pair
        unsigned *d_keys = nullptr;
        unsigned *d_values = nullptr;
        cudaMalloc((void **) &d_values, sizeof(unsigned) * numDense);
        TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");
        // prepare the input hash
        arangeKernel<<<tv::launch::getBlocks(numDense), tv::launch::CUDA_NUM_THREADS, 0,
        d.getStream()>>>(d_values, numDense);
        TV_CHECK_CUDA_ERR_V2("assignKernel failed");

        cudaMalloc((void **) &d_keys, sizeof(unsigned) * numDense);
        TV_CHECK_CUDA_ERR_V2("cudaMalloc failed");

        arangeIndexKernel<<<tv::launch::getBlocks(numDense), tv::launch::CUDA_NUM_THREADS, 0,
        d.getStream()>>>(d_keys, tv::torch2tv<const int>(ritensor), numDense, rss[0], rss[1], rss[2]);
        TV_CHECK_CUDA_ERR_V2("assignIndexKernel failed");

        bool res = table.Build(numDense, d_keys, d_values);
        while (!res and initial_size < 4.) {
            initial_size += 0.5;
            table.Initialize(numDense, initial_size, 4);
            res = table.Build(numDense, d_keys, d_values);
        }
        TV_ASSERT_RT_ERR(res, "Table Build failed")

        cudaFree(d_keys);
        cudaFree(d_values);

        auto tableSize = table.get_table_size();
        auto tableData = table.data();
        auto constants = table.get_constants_4();
        auto stash_constants = table.get_stash_constants();
        auto stash_count = table.get_stash_count();

        //////////////////////////////////////

        auto kernel_idx = torch::full({2, kernel_size[0] * kernel_size[1], numOut}, -1,
                                      torch::dtype(torch::kInt32).device(qitensor.device()));

        auto kernel_number = torch::zeros({kernel_size[0] * kernel_size[1]},
                                          torch::dtype(torch::kInt32).device(qitensor.device()));

        frust_conv_kernel<<<tv::launch::getBlocks(
                numOut),
        tv::launch::CUDA_NUM_THREADS, 0, d.getStream()>>>(
                tv::torch2tv<const float>(qgtensor),
                tv::torch2tv<const int>(qitensor),
                tv::torch2tv<const float>(qrange),
                tv::torch2tv<const float>(rgtensor),
                tv::torch2tv<const float>(rrange),
                tv::torch2tv<const int>(idx),
                tv::torch2tv<int>(kernel_idx),    // (2,K*K,N) // initial -1,-1
                tv::torch2tv<int>(kernel_number), // (K*K,)
                flag,
                distance,
                kernel_size,
                dilation,
                rss, // include (H,W,KM)
                tableSize,
                tableData,
                constants,
                stash_constants,
                stash_count);
        return {kernel_idx, kernel_number};
    }


} // namespace spproj