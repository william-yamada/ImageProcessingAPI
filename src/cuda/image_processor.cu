#include "image_processor.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int ix = x + kx;
                int iy = y + ky;
                sum += input[iy * width + ix];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(sum / 9);
    }
}

bool cuda_available() {
    int count;
    cudaGetDeviceCount(&count);
    return count > 0;
}

std::vector<unsigned char> CUDAImageProcessor::process(const std::vector<unsigned char>& image, int width, int height) {
    std::vector<unsigned char> output(image.size(), 0);
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, image.size());
    cudaMalloc(&d_output, image.size());
    cudaMemcpy(d_input, image.data(), image.size(), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    blur_kernel<<<blocks, threads>>>(d_input, d_output, width, height);

    cudaMemcpy(output.data(), d_output, image.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}