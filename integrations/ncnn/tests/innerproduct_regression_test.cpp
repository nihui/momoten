// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "testutil.h"

static int run_innerproduct(int w, int h, int outch, bool use_fp16_packed,
                            bool use_shader_local_memory)
{
    ncnn::Mat input = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 1);
    pd.set(2, outch * w);
    pd.set(8, 1);
    pd.set(9, 0);
    pd.set(10, ncnn::Mat());

    std::vector<ncnn::Mat> weights(4);
    weights[0] = RandomMat(outch * w);
    weights[1] = RandomMat(outch);
    weights[2] = scales_mat(weights[0], outch, w, w);
    weights[3] = scales_mat(input, 1, w, w);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = use_fp16_packed;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_shader_local_memory = use_shader_local_memory;

    const int ret = test_layer_opt("InnerProduct", pd, weights, opt, input, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "ncnn_innerproduct_regression_test: w=%d h=%d outch=%d fp16_packed=%d local_memory=%d failed\n",
                w, h, outch, use_fp16_packed ? 1 : 0,
                use_shader_local_memory ? 1 : 0);
    }
    return ret;
}

int main()
{
    SRAND(7767517);
    if (run_innerproduct(4, 15, 8, false, false) != 0)
        return 1;

    SRAND(7767517);
    if (run_innerproduct(4, 15, 8, false, true) != 0)
        return 1;

    SRAND(7767517);
    if (run_innerproduct(9, 8, 7, true, false) != 0)
        return 1;

    SRAND(7767517);
    if (run_innerproduct(9, 8, 7, true, true) != 0)
        return 1;

    return 0;
}
