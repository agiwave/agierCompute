cmake_minimum_required(VERSION 3.15)

project(ace_core VERSION 1.0.0 LANGUAGES C)

add_library(ace_core SHARED engine.c)

target_include_directories(ace_core 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(ace_core PRIVATE ACE_CORE_EXPORTS)

if(MSVC)
    target_compile_options(ace_core PRIVATE /W3)
else()
    target_compile_options(ace_core PRIVATE -Wall -Wextra)
endif()

# Install
install(TARGETS ace_core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
