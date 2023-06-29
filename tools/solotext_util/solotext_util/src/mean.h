//
// Created by hp on 2023/4/4.
//

#ifndef MEAN_H
#define MEAN_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

// std::tuple<torch::Tensor, torch::Tensor> meaning(const torch::Tensor &label, const torch::Tensor &kernel, const torch::Tensor &column_unique);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
meaning(const torch::Tensor &label,
        const torch::Tensor &kernel,
        const torch::Tensor &column_unique,
        const torch::Tensor &kernel_index);

#endif 
