#include "image_processor.hpp"
#include <iostream>

#ifdef USE_CUDA
#include "cuda/cuda_image_processor.cuh"
#endif

namespace {
    void apply_blur_cpu(std::vector<unsigned char>& out, const std::vector<unsigned char>& in, int width, int height) {
        const int kernel[3][3] = {
            {1, 1, 1},
            {1, 1, 1},
            {1, 1, 1}
        };
        const int kernel_sum = 9;

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int px = x + kx;
                        int py = y + ky;
                        sum += in[py * width + px] * kernel[ky + 1][kx + 1];
                    }
                }
                out[y * width + x] = static_cast<unsigned char>(sum / kernel_sum);
            }
        }
    }
}

class CPUImageProcessor : public ImageProcessor {
public:
    std::vector<unsigned char> process(const std::vector<unsigned char>& image, int width, int height) override {
        std::vector<unsigned char> output(image.size(), 0);
        apply_blur_cpu(output, image, width, height);
        return output;
    }
};

ImageProcessor* create_processor() {
#ifdef USE_CUDA
    if (cuda_available()) return new CUDAImageProcessor();
#endif
    return new CPUImageProcessor();
}