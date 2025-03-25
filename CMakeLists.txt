cmake_minimum_required(VERSION 3.22)
project(ImageProcessorAPI LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/json)

set(SOURCES
    src/main.cpp
    src/image_processor.cpp
    src/cuda/image_processor.cu
    api/server.cpp
    api/routes.cpp
)

# Add executable
add_executable(image_processor_api ${SOURCES})

set_target_properties(image_processor_api PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89    
)

target_link_libraries(image_processor_api
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    pthread=
)