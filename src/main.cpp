#include "image_processor.hpp"
#include <fstream>
#include <iostream>

int main() {
    ImageProcessor* processor = create_processor();
    std::vector<unsigned char> image(1024 * 768, 128); // Dummy grayscale image
    auto result = processor->process(image, 1024, 768);
    std::cout << "Processed image with " << result.size() << " pixels.\n";
    delete processor;
    return 0;
}