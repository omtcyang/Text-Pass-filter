#ifndef _CUDA_HEADER_H_
#define _CUDA_HEADER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor clustering(const torch::Tensor &input);

#endif
