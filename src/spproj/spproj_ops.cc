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

#include <spproj/spprojops.h>

namespace spproj {

    std::vector <torch::Tensor> spherical_projection_index(torch::Tensor xyz,
                                                           torch::Tensor itensor2d,
                                                           std::vector <int64_t> spatial_size
    ) {
    /* used for spherical frustum structure construction */
        TV_ASSERT_RT_ERR(xyz.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(itensor2d.is_cuda(), "tensor not on CUDA")


        torch::Tensor batchinfo = itensor2d.index({"...", 2});
        torch::Tensor icol = itensor2d.index({"...", 0});
        torch::Tensor irow = itensor2d.index({"...", 1});

        torch::Tensor idx = (batchinfo * (spatial_size[0] * spatial_size[1]) + irow * spatial_size[1] + icol).toType(
                torch::kInt32);

        torch::Tensor im = torch::empty_like(idx);
        torch::Tensor xi = torch::ones({idx.size(0), 1}, torch::dtype(torch::kFloat32).device(idx.device()));

        // logn
        auto sorted_return = torch::sort(idx);
        torch::Tensor sorted = std::get<0>(sorted_return);
        torch::Tensor sorted_idx = std::get<1>(sorted_return).toType(torch::kInt32);

        spherical_assign(tv::GPU(), tv::torch2tv<float>(xi), tv::torch2tv<int>(im),
                        tv::torch2tv<const int>(sorted), tv::torch2tv<const int>(sorted_idx));

        torch::Tensor gtensor = torch::cat({xyz, xi}, 1);
        torch::Tensor itensor = torch::stack({icol, irow, im, batchinfo}, 1);

        return {itensor, gtensor};
    }

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
            std::vector <int64_t> spatial_size) // [H,W,M]
    /** return {sampled_itensor, sampled_gtensor, sample_idx} **/
    /** new spatial size=(ceil(H/stride[0]),ceil(W/stride[1]) because of the u%stride==0 sampled **/\
    {
        TV_ASSERT_RT_ERR(rgtensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(ritensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(qgtensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(qitensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(idx.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(random_hw.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(out_index.is_cuda(), "tensor not on CUDA")

        float dist = (float) distance;
        // check the input metainfo
        TV_ASSERT_RT_ERR(dstride.size() == 2, "Must be 2 dim stride (u,v).")
        TV_ASSERT_RT_ERR(qstride.size() == 2, "Must be 2 dim stride (u,v).")
        TV_ASSERT_RT_ERR(kernel_size.size() == 2, "Must be 2 dim kernel.")
        TV_ASSERT_RT_ERR(spatial_size.size() == 3, "Must be 3 dim spatial size.")

        int64_t spatial_volume = spatial_size[0] * spatial_size[1] * spatial_size[2];

        TV_ASSERT_RT_ERR(batch_size * spatial_volume < std::numeric_limits<int>::max(),
                         "Over the hash limit") // 2**32, normally not over
        // change the input tensor and vector to tv for cuda process
        tv::SimpleVector<int, 2> dstride_tv;
        tv::SimpleVector<int, 2> qstride_tv;
        tv::SimpleVector<int, 2> kernel_size_tv;
        tv::SimpleVector<int, 3> ss_tv;

        for (int i = 0; i < 2; ++i) {
            kernel_size_tv.push_back(kernel_size[i]);
            qstride_tv.push_back(qstride[i]);
            dstride_tv.push_back(dstride[i]);
        }
        for (int i = 0; i < 3; ++i) {

            ss_tv.push_back(spatial_size[i]);
        }

        /* all tensor build in cu*/

        return rulebook_query_frustfps(tv::TorchGPU(), qitensor, qgtensor, idx, ritensor, rgtensor,
                                       tv::torch2tv<const int>(out_index), tv::torch2tv<const int>(random_hw),
                                       dstride_tv, qstride_tv, kernel_size_tv, ss_tv, dist, flag, neighbor_number
        );
    }

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
            std::vector <int64_t> spatial_size)
          /*
* This is the main function of spherical frustum sparse convolution (SFC)
* Inputs:
* The query qgtensor xyzf [N,4] float
* The query qitensor uvmb [N,4] int
* The reference rgtensor xyzf [M,4] float
* The reference ritensor uvmb [M,4] int
** We build the output hash map after input this will take O(N) time. The input hash map will not be built.
* Batch size: [INT]
*/
             {
        TV_ASSERT_RT_ERR(rgtensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(ritensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(qgtensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(qitensor.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(idx.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(qrange.is_cuda(), "tensor not on CUDA")
        TV_ASSERT_RT_ERR(rrange.is_cuda(), "tensor not on CUDA")

        float dist = (float) distance;
        // check the input metainfo
        TV_ASSERT_RT_ERR(dilation.size() == 2, "Must be 2 dim dilation (u,v).")
        TV_ASSERT_RT_ERR(kernel_size.size() == 2, "Must be 2 dim kernel.")
        TV_ASSERT_RT_ERR(spatial_size.size() == 3, "Must be 3 dim spatial size.")

        int64_t spatial_volume = spatial_size[0] * spatial_size[1] * spatial_size[2];

        TV_ASSERT_RT_ERR(batch_size * spatial_volume < std::numeric_limits<int>::max(),
                         "Over the hash limit") // 2**32, normally not over
        tv::SimpleVector<int, 2> kernel_size_tv;
        tv::SimpleVector<int, 2> dilation_tv;
        tv::SimpleVector<int, 3> ss_tv;

        for (int i = 0; i < 2; ++i) {
            kernel_size_tv.push_back(kernel_size[i]);
            dilation_tv.push_back(dilation[i]);
        }
        for (int i = 0; i < 3; ++i) {

            ss_tv.push_back(spatial_size[i]);
        }


        return rulebook_query_frustconv(tv::TorchGPU(), qitensor, qrange, qgtensor, idx,
                                        ritensor, rgtensor, rrange,
                                        kernel_size_tv,
                                        ss_tv,
                                        dilation_tv,
                                        dist,
                                        flag);
    }

} // namespace spproj