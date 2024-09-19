cmake_minimum_required(VERSION 3.25)
project(COSC-4P80-Assignment-1 VERSION 0.0.4)

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/blt)

include_directories(include/)
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(COSC-4P80-Assignment-1 ${PROJECT_BUILD_FILES})

target_compile_options(COSC-4P80-Assignment-1 PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-comment)
target_link_options(COSC-4P80-Assignment-1 PRIVATE -Wall -Wextra -Werror -Wpedantic -Wno-comment)

target_link_libraries(COSC-4P80-Assignment-1 PRIVATE BLT)

if (${ENABLE_ADDRSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-1 PRIVATE -fsanitize=address)
    target_link_options(COSC-4P80-Assignment-1 PRIVATE -fsanitize=address)
endif ()

if (${ENABLE_UBSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-1 PRIVATE -fsanitize=undefined)
    target_link_options(COSC-4P80-Assignment-1 PRIVATE -fsanitize=undefined)
endif ()

if (${ENABLE_TSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Assignment-1 PRIVATE -fsanitize=thread)
    target_link_options(COSC-4P80-Assignment-1 PRIVATE -fsanitize=thread)
endif ()
