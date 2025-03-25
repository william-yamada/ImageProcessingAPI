#ifndef CUDA_IMAGE_PROCESSOR_CUH
#define CUDA_IMAGE_PROCESSOR_CUH

#include "image_processor.hpp"

bool cuda_available();

class CUDAImageProcessor : public ImageProcessor {
public:
    std::vector<unsigned char> process(const std::vector<unsigned char>& image, int width, int height) override;
};

#endif