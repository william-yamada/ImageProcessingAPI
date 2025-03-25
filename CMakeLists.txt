cmake_minimum_required(VERSION 3.22)
project(ImageProcessorAPI LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_LINK_EXECUTABLE "${CMAKE_CUDA_LINK_EXECUTABLE} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart shared")


# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
include(FetchContent)

FetchContent_Declare(
    crow
    GIT_REPOSITORY https:///github.com/CrowCpp/Crow.git
    GIT_TAG v1.0+1
)
set(CROW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CROW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CROW_USE_SYSTEM_ASIO OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(crow)

FetchContent_Declare(
    json
    GIT_REPOSITORY https:///github.com/nlohmann/json
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    # src/main.cpp
    src/image_processor.cpp
    src/cuda/image_processor.cu
    api/server.cpp
    api/routes.cpp
)

# Add executable
add_executable(image_processor_api ${SOURCES})

set_target_properties(image_processor_api PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 52 60 61 75 86
)

target_link_libraries(image_processor_api
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    Crow::Crow
    nlohmann_json::nlohmann_json
)