#include "image_processor.hpp"
#include <iostream>

#ifdef USE_CUDA
#include "cuda/cuda_image_processor.cuh"
#endif

class CPUImageProcessor : public ImageProcessor {
public:
    std::vector<unsigned char> process(const std::vector<unsigned char>& image, int width, int height) override {
        std::vector<unsigned char> output(image);
        for (auto& pixel : output) pixel = 255 - pixel;
        return output;
    }
};

ImageProcessor* create_processor() {
#ifdef USE_CUDA
    if (cuda_available()) return new CUDAImageProcessor();
#endif
    return new CPUImageProcessor();
}
