// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "testutil.h"

static int run_convolution(bool use_shader_local_memory)
{
    const int w = 9;
    const int h = 7;
    const int c = 16;
    const int outch = 16;
    const int kernel = 3;

    ncnn::Mat input = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, 1);
    pd.set(5, 0);
    pd.set(6, outch * c * kernel * kernel);
    pd.set(9, 0);
    pd.set(10, ncnn::Mat());

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(outch * c * kernel * kernel);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_sgemm_convolution = true;
    opt.use_winograd_convolution = true;
    opt.use_shader_local_memory = use_shader_local_memory;

    const int ret = test_layer_opt("Convolution", pd, weights, opt, input, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "ncnn_convolution_regression_test: local_memory=%d failed\n",
                use_shader_local_memory ? 1 : 0);
    }
    return ret;
}

static int run_fp16_packed_convolution(bool use_shader_local_memory)
{
    const int w = 9;
    const int h = 7;
    const int c = 8;
    const int outch = 13;

    ncnn::Mat input = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, 1);
    pd.set(2, 1);
    pd.set(3, 2);
    pd.set(4, 0);
    pd.set(5, 0);
    pd.set(6, outch * c);
    pd.set(9, 0);
    pd.set(10, ncnn::Mat());

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(outch * c);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_sgemm_convolution = true;
    opt.use_winograd_convolution = true;
    opt.use_shader_local_memory = use_shader_local_memory;

    const int ret = test_layer_opt("Convolution", pd, weights, opt, input, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "ncnn_convolution_regression_test: fp16_packed=1 local_memory=%d failed\n",
                use_shader_local_memory ? 1 : 0);
    }
    return ret;
}

int main()
{
    SRAND(7767517);
    if (run_convolution(false) != 0)
        return 1;

    SRAND(7767517);
    if (run_convolution(true) != 0)
        return 1;

    SRAND(7767517);
    if (run_fp16_packed_convolution(false) != 0)
        return 1;

    SRAND(7767517);
    if (run_fp16_packed_convolution(true) != 0)
        return 1;

    return 0;
}
