//
// Created by hp on 2023/4/7.
//

#ifndef GET_TEXT_MASKS_H
#define GET_TEXT_MASKS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

std::tuple<torch::Tensor, torch::Tensor>
get_masks_counter(const torch::Tensor &mask_connect_labels,
                  const torch::Tensor &all_text_labels);


#endif
