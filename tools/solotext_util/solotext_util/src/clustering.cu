////
//// Created by hp on 2023/3/28.
////
//
#include "clustering.h"
//#include <iostream>

__global__ void labeling(bool *ronghe, const bool *input, const uint16_t W, const uint16_t H) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W)return;
    int first = 1;

    for (int i = 0; i < H; i++) {
        if (input[i * W + idx]) {
            if (first == 1) {
                for (int j = 0; j < W; j++) {
                    unsigned long long now = input[i * W + j];
                    ronghe[idx * W + j] = now;
                }
                first = 0;
            } else {
                for (int j = 0; j < W; j++) {
                    unsigned long long now = input[i * W + j];
                    ronghe[idx * W + j] &= now;
                }
            }
        }
    }
}


__global__ void final_labeling(int *label, const bool *ronghe, const uint16_t W, const uint16_t H) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W) return;
    for (int i = 0; i < H; i++) {
        if (label[idx] == 0) {
            if (ronghe[i * W + idx])
                label[idx] = i + 1;
        } else {
            return;
        }
    }
}


torch::Tensor clustering(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    const uint16_t H = input.size(-2);
    const uint16_t W = input.size(-1);
    auto label_options = torch::TensorOptions().dtype<bool>().device(input.device());

    torch::Tensor ronghe = torch::zeros({W, W}, label_options);

    int block_size = 32;
    int grid_size = (W + block_size - 1) / block_size;
    labeling<<<grid_size, block_size>>>(ronghe.data_ptr<bool>(), input.data_ptr<bool>(), W, H);


    torch::Tensor label = torch::zeros(W, label_options.dtype<int>());

//    auto ronghe2 = ronghe.to(torch::kCPU);

//    std::set<int> cache;
//    for (int j = 0; j < W; ++j) {
//        cache.insert(j);
//    }
//    for (int i = 0; i < H; ++i) {
//
//        for (int j = 0; j < W; ++j) {
////        while (!cache.empty()) {
////            int j = *cache.begin();
//            if (label.index({j}).item<int>() == 0 && ronghe.index({i, j}).item<bool>()) {
//                label.index_put_({j}, i + 1);
////                cache.erase(j);
//            }
////        }
//        }
//    }


    final_labeling<<<grid_size, block_size>>>(label.data_ptr<int>(), ronghe.data_ptr<bool>(), W, W);
//    int label_index = 1;

//    auto cmp = [](const at::Tensor &a, const at::Tensor &b) {
//        int maxIndex = max(a.size(0), b.size(0));
//
//    };
//
//    for (int i = 0; i < W; i++) {
//        if (label[i] == 0) {
//            at::Tensor temp = ronghe[i];
//            label.index_put_({temp}, label_index);
//            label_index++;
//        }
//    }

//    at::Tensor b = input.index({"...", zero_index});
//    at::Tensor select_tensor = input.index({b});
//    at::Tensor ronghe = torch::sum(select_tensor, 0) >= select_tensor.size(-2);
//    label.index_put_({ronghe}, label_index);
//    label_index += 1;
//    auto sum = torch::sum(ronghe, 0).item().toInt();

    return label;
}
