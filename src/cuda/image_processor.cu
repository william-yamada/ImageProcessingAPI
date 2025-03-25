#include "image_processor.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void invert_kernel(unsigned char* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 255 - data[idx];
}

bool cuda_available() {
    int count;
    cudaGetDeviceCount(&count);
    return count > 0;
}

std::vector<unsigned char> CUDAImageProcessor::process(const std::vector<unsigned char>& image, int width, int height) {
    std::vector<unsigned char> output(image.size());
    unsigned char* d_image;
    cudaMalloc(&d_image, image.size());
    cudaMemcpy(d_image, image.data(), image.size(), cudaMemcpyHostToDevice);
    invert_kernel<<<(image.size()+255)/256, 256>>>(d_image, image.size());
    cudaMemcpy(output.data(), d_image, image.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    return output;
}
