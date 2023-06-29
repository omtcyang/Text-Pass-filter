#include "buf.h"

// 3d
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

namespace cc3d
{
    __global__ void init_labeling(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D) {
        const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z);

        const uint32_t idx = z * W * H + y * W + x;

        if (x < W && y < H && z < D)
            label[idx] = idx;
    }

    __global__ void merge(uint8_t * const img, int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z);
        const uint32_t idx = z * W * H + row * W + col;

        if (row >= H || col >= W || z >= D)
            return;

        uint32_t P = 0;

        if (img[idx])                      P |= 0x777;
        if (row + 1 < H && img[idx + W])   P |= 0x777 << 4;
        if (col + 1 < W && img[idx + 1])   P |= 0x777 << 1;

        if (col == 0)               P &= 0xEEEE;
        if (col + 1 >= W)           P &= 0x3333;
        else if (col + 2 >= W)      P &= 0x7777;

        if (row == 0)               P &= 0xFFF0;
        if (row + 1 >= H)           P &= 0xFF;

        if (P > 0)
        {
            // If need check about top-left pixel(if flag the first bit) and hit the top-left pixel
            if (hasBit(P, 0) && img[idx - W - 1]){
                union_(label, idx, idx - 2 * W - 2); // top left block
            }

            if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                union_(label, idx, idx - 2 * W); // top bottom block

            if (hasBit(P, 3) && img[idx + 2 - W])
                union_(label, idx, idx - 2 * W + 2); // top right block

            if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                union_(label, idx, idx - 2); // just left block
        }
    }

    __global__ void compression(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z);

        const uint32_t idx = z * W * H + y * W + x;

        if (x < W && y < H && z < D)
            find_n_compress(label, idx);
    }


    __global__ void final_labeling(const uint8_t *img, int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D)
    {


        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z);

        const uint32_t idx = z * W * H + row * W + col;

        if (row >= H || col >= W || z >= D)
            return;

        int32_t y = label[idx] + 1;

        if (img[idx])
            label[idx] = y;
        else
            label[idx] = 0;

        if (col + 1 < W)
        {
            if (img[idx + 1])
                label[idx + 1] = y;
            else
                label[idx + 1] = 0;

            if (row + 1 < H)
            {
                if (img[idx + W + 1])
                    label[idx + W + 1] = y;
                else
                    label[idx + W + 1] = 0;
            }
        }

        if (row + 1 < H)
        {
            if (img[idx + W])
                label[idx + W] = y;
            else
                label[idx + W] = 0;
        }
    } // final_labeling
} // namespace cc3d


torch::Tensor connected_componnets_labeling_3d(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 3, "input must be a  [D, H, W] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t D = input.size(-3);
    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);

    // AT_ASSERTM((D % 2) == 0, "shape must be a even number");
    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor label = torch::zeros({D, H, W}, label_options);

    dim3 grid = dim3(((W + 1) / 2 + BLOCK_X - 1) / BLOCK_X, ((H + 1) / 2 + BLOCK_Y - 1) / BLOCK_Y, (D + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    
    cc3d::init_labeling<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), W, H, D
    );
    cc3d::merge<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        W, H, D
    );
    cc3d::compression<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), W, H, D
    );
    cc3d::final_labeling<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        W, H, D
    );
    return label;
}
