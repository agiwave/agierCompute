cmake_minimum_required(VERSION 3.15)
project(AgierCompute VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add modules - each module has its own CMakeLists.txt
add_subdirectory(src/core)
add_subdirectory(src/cpu)
add_subdirectory(src/cuda)
add_subdirectory(src/opencl)
add_subdirectory(src/vulkan)
add_subdirectory(examples)

# Print info
message(STATUS "=== AgierCompute Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")