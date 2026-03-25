cmake_minimum_required(VERSION 3.15)

project(ace_be_cpu VERSION 1.0.0 LANGUAGES C)

add_library(ace_be_cpu SHARED cpu_backend.c)

target_include_directories(ace_be_cpu 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(ace_be_cpu PRIVATE ACE_BACKEND_EXPORTS)
target_link_libraries(ace_be_cpu PRIVATE ace_core)

if(MSVC)
    target_compile_options(ace_be_cpu PRIVATE /W3 /D_CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(ace_be_cpu PRIVATE -Wall -Wextra)
endif()

install(TARGETS ace_be_cpu LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
