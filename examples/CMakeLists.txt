cmake_minimum_required(VERSION 3.15)

project(ace_examples VERSION 1.0.0 LANGUAGES C)

# 简单测试
add_executable(simple_test simple_test.c)
target_include_directories(simple_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(simple_test PRIVATE ace_core)
set_target_properties(simple_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Linux RPATH
if(NOT WIN32)
    set_target_properties(simple_test PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()

install(TARGETS simple_test RUNTIME DESTINATION bin)
