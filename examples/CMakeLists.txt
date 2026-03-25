cmake_minimum_required(VERSION 3.15)

project(ace_examples VERSION 1.0.0 LANGUAGES C CXX)

# Simple test
add_executable(simple_test simple_test.c)

# Demo (C++)
add_executable(ace_demo demo.cpp)

target_include_directories(ace_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ace_demo PRIVATE ace_core)

# Copy backend DLLs to demo directory for testing
if(WIN32)
    add_custom_command(TARGET ace_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:ace_be_cpu>"
        "$<TARGET_FILE_DIR:ace_demo>"
    )
    add_custom_command(TARGET ace_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:ace_be_opencl>"
        "$<TARGET_FILE_DIR:ace_demo>"
    )
endif()

install(TARGETS ace_demo RUNTIME DESTINATION bin)