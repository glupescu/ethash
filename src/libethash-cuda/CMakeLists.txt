cmake_minimum_required(VERSION 2.8)

set(LIBRARY ethash-cuda)
set(CMAKE_BUILD_TYPE Release)

include_directories(/export/home/fils/stud/g/grigore.lupescu/ethereum/cryptopp)

if (NOT MSVC)
	# Initialize CXXFLAGS for c++11
    set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
    set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
    set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

    # Compiler-specific C++11 activation.
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
            message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
        endif ()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    else ()
        message(FATAL_ERROR "Your C++ compiler does not support C++11.")
    endif ()
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

if (NOT CUDA_FOUND)
	find_package(CUDA)
endif()

if (CUDA_FOUND)
	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -pedantic -fPIC ${CMAKE_CXX_FLAGS}")
	include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
	include_directories(..)
	CUDA_ADD_LIBRARY(${LIBRARY} ethash_cuda_miner.cpp ethash_cuda_miner.h)
	TARGET_LINK_LIBRARIES(${LIBRARY} ${CUDA_LIBRARIES} ethash)
endif()
