#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <vector>
#include <string>

class ImageProcessor {
public:
    virtual std::vector<unsigned char> process(const std::vector<unsigned char>& image, int width, int height) = 0;
    virtual ~ImageProcessor() = default;
};

ImageProcessor* create_processor();

#endif // IMAGE_PROCESSOR_HPP