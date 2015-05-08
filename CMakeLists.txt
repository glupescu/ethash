cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_C_COMPILER /opt/tools/compilers/gnu/4.7.0/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/tools/compilers/gnu/4.7.0/bin/g++)

project(ethash)

set(CRYPTOPP_INCLUDE_DIR /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/cryptopp CACHE FILEPATH "" FORCE)
set(CRYPTOPP_ROOT_DIR /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/cryptopp CACHE FILEPATH "" FORCE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(ETHHASH_LIBS ethash)

include_directories(/opt/tools/libraries/cuda-6.0/include/)
include_directories(/export/home/fils/stud/g/grigore.lupescu/ethereum-wp/cryptopp/)
include_directories(/export/home/fils/stud/g/grigore.lupescu/ethereum-wp/)

if (WIN32 AND WANT_CRYPTOPP)
	add_subdirectory(cryptopp)
endif()

add_subdirectory(src/libethash)

if (NOT CUDA_FOUND)
	find_package(CUDA)
endif()

if (CUDA_FOUND)
	add_subdirectory(src/libethash-cuda)
endif()

add_subdirectory(src/benchmark EXCLUDE_FROM_ALL)
add_subdirectory(test/c)
