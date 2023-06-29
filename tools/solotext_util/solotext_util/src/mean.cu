// //
// // Created by hp on 2023/4/4.
// //

// #include "mean.h"
// #include "time.h"
// #include <iostream>
// #include <tuple>

// __global__ void
// meaning_(const int *label, const float *kernel, const int *column_unique,
//          float *sum, int *counter, int kernel_W, int label_count, int unique_count) {
//     const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= unique_count)return;
//     for (int i = 0; i < label_count; ++i) {
//         if (label[i] == column_unique[idx]) {
//             for (int j = 0; j < kernel_W; j++)
//                 sum[idx * kernel_W + j] += kernel[i * kernel_W + j];
//             counter[idx] += 1;
//         }
//     }
// }

// std::tuple<torch::Tensor, torch::Tensor>
// meaning(const torch::Tensor &label, const torch::Tensor &kernel, const torch::Tensor &column_unique) {
//     AT_ASSERTM(label.is_cuda(), "label must be a CUDA tensor");
//     AT_ASSERTM(kernel.is_cuda(), "kernel must be a CUDA tensor");
//     AT_ASSERTM(label.size(-1) == kernel.size(-2), "label size must equal to kernel size");
//     auto label_options = torch::TensorOptions().dtype<float>().device(kernel.device());
//     int unique_count = column_unique.size(-1);

//     torch::Tensor sum = torch::zeros({unique_count, kernel.size(-1)}, label_options);
//     torch::Tensor counter = torch::zeros({unique_count}, label_options.dtype<int>());


//     int block_size = 32;
//     int grid_size = (unique_count + block_size - 1) / block_size;
//     meaning_<<<grid_size, block_size>>>(label.data_ptr<int>(),
//                                         kernel.data_ptr<float>(),
//                                         column_unique.data_ptr<int>(),
//                                         sum.data_ptr<float>(),
//                                         counter.data_ptr<int>(),
//                                         kernel.size(-1),
//                                         label.size(-1),
//                                         unique_count);
//     return std::make_tuple(sum, counter);
// //    return sum;
// }


#include "mean.h"
#include <tuple>

__global__ void
meaning_(const int *label, const float *kernel, const int *column_unique, const long *kernel_index,
         float *sum, int *counter, long *HW, int kernel_W, int label_count, int unique_count) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= unique_count)return;
    int index = 0;
    for (int i = 0; i < label_count; ++i) {
        if (label[i] == column_unique[idx]) {
            for (int j = 0; j < kernel_W; j++)
                sum[idx * kernel_W + j] += kernel[i * kernel_W + j];
            counter[idx] += 1;
            HW[idx * label_count + index] = kernel_index[i];
            index++;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
meaning(const torch::Tensor &label,
        const torch::Tensor &kernel,
        const torch::Tensor &column_unique,
        const torch::Tensor &kernel_index) {
    AT_ASSERTM(label.is_cuda(), "label must be a CUDA tensor");
    AT_ASSERTM(kernel.is_cuda(), "kernel must be a CUDA tensor");
    AT_ASSERTM(label.size(-1) == kernel.size(-2), "label size must equal to kernel size");
    auto label_options = torch::TensorOptions().dtype<float>().device(kernel.device());
    int unique_count = column_unique.size(-1);

    torch::Tensor sum = torch::zeros({unique_count, kernel.size(-1)}, label_options);
    torch::Tensor counter = torch::zeros({unique_count}, label_options.dtype<int>());
    torch::Tensor HW = torch::zeros({unique_count, label.size(-1)}, label_options.dtype<long>());


    int block_size = 32;
    int grid_size = (unique_count + block_size - 1) / block_size;
    meaning_<<<grid_size, block_size>>>(label.data_ptr<int>(),
                                        kernel.data_ptr<float>(),
                                        column_unique.data_ptr<int>(),
                                        kernel_index.data_ptr<long>(),
                                        sum.data_ptr<float>(),
                                        counter.data_ptr<int>(),
                                        HW.data_ptr<long>(),
                                        kernel.size(-1),
                                        label.size(-1),
                                        unique_count);
    return std::make_tuple(sum, counter, HW);
}
