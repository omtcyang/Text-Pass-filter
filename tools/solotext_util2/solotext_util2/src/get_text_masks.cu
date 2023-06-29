//
// Created by hp on 2023/4/7.
//

#include "get_text_masks.h"


#define BLOCK_ROWS 16
#define BLOCK_COLS 16


__global__ void per_mask(const int *mask_connect_label,
                         int *res_mask,
                         int H,
                         int W,
                         int label) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    if (mask_connect_label[y * W + x] == label)
        res_mask[y * W + x] = 1;
}


__global__ void get_masks_counter_(int *mask_connect_labels,
                                   const int *all_text_labels,
                                   int *res_masks,
                                   int *res_counter,
                                   int mask_counts,
                                   int mask_H,
                                   int mask_W,
                                   int all_text_labels_W) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mask_counts)return;
    int last = -1;
    int now_index = 0;
    int pre = idx * mask_counts * all_text_labels_W + idx * all_text_labels_W;
    dim3 grid = dim3((mask_W + BLOCK_COLS - 1) / BLOCK_COLS, (mask_H + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);
    int *mask_connect_label = mask_connect_labels + idx * mask_H * mask_W;


    for (int i = 0; i < all_text_labels_W; i++) {
        if (all_text_labels[pre + i] == 0)
            break;
        if (all_text_labels[pre + i] == last)
            continue;
        last = all_text_labels[pre + i];
        int *res_mask = res_masks + ((idx * 50 + now_index) * mask_H * mask_W);
        per_mask<<<grid, block>>>(mask_connect_label, res_mask, mask_H, mask_W, last);
        now_index += 1;
        res_counter[idx] += 1;
    }
}


// __global__ void get_masks_counter_(const int *mask_connect_labels,
//                                    const int *all_text_labels,
//                                    int *res_masks,
//                                    int *res_counter,
//                                    int mask_counts,
//                                    int mask_H,
//                                    int mask_W,
//                                    int all_text_labels_W) {
//     const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= mask_counts)return;
//     int last = -1;
//     int now_index = 0;
//     int pre = idx * mask_counts * all_text_labels_W + idx * all_text_labels_W;
//     for (int i = 0; i < all_text_labels_W; i++) {
//         if (all_text_labels[pre + i] == 0)
//             break;
//         if (all_text_labels[pre + i] == last)
//             continue;
//         last = all_text_labels[pre + i];
//         for (int j = 0; j < mask_H; j++) {
//             for (int k = 0; k < mask_W; k++) {
//                 if (mask_connect_labels[idx * mask_H * mask_W + j * mask_W + k] == last)
//                     res_masks[(idx * 50 + now_index) * mask_H * mask_W + j * mask_W + k] = 1;
//             }
//         }
//         now_index += 1;
//         res_counter[idx] += 1;
//     }
// }

std::tuple<torch::Tensor, torch::Tensor>
get_masks_counter(const torch::Tensor &mask_connect_labels,
                  const torch::Tensor &all_text_labels) {
    AT_ASSERTM(mask_connect_labels.is_cuda(), "mask_connect_labels must be a CUDA tensor");
    AT_ASSERTM(all_text_labels.is_cuda(), "all_text_labels must be a CUDA tensor");
    AT_ASSERTM(mask_connect_labels.size(-3) == all_text_labels.size(-3), "size must be same");
    AT_ASSERTM(mask_connect_labels.size(-3) == all_text_labels.size(-2), "size must be same");
    auto label_options = torch::TensorOptions().dtype<int>().device(mask_connect_labels.device());
    int mask_counts = mask_connect_labels.size(-3);
    int mask_H = mask_connect_labels.size(-2);
    int mask_W = mask_connect_labels.size(-1);
    torch::Tensor res_masks = torch::zeros({mask_counts, 50, mask_H, mask_W}, label_options);
    torch::Tensor res_counter = torch::zeros({mask_counts}, label_options);
    int block_size = 32;
    int grid_size = (mask_counts + block_size - 1) / block_size;
    get_masks_counter_<<<grid_size, block_size>>>(mask_connect_labels.data_ptr<int>(),
                                                  all_text_labels.data_ptr<int>(),
                                                  res_masks.data_ptr<int>(),
                                                  res_counter.data_ptr<int>(),
                                                  mask_counts,
                                                  mask_H,
                                                  mask_W,
                                                  all_text_labels.size(-1));
    return std::make_tuple(res_masks, res_counter);
}

