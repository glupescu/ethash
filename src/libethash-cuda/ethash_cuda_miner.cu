/*
  This file is part of c-ethash.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file ethash_cuda_miner.cpp
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*/


#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <queue>
#include <vector>
#include <libethash/util.h>
#include <string>
#include "ethash_cuda_miner.h"

#include <cuda_runtime_api.h>
#include <cuda.h>

#define ETHASH_BYTES 32

#undef min
#undef max

// Kernel that executes on the CUDA device
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

static void add_definition(std::string& source, char const* id, unsigned value)
{

}

ethash_cuda_miner::ethash_cuda_miner()
{
}

void ethash_cuda_miner::finish()
{
}

bool ethash_cuda_miner::init(ethash_params const& params, ethash_h256_t const *seed, unsigned workgroup_size)
{
	   int n = 100000;

	    // Host input vectors
	    double *h_a;
	    double *h_b;
	    //Host output vector
	    double *h_c;

	    // Device input vectors
	    double *d_a;
	    double *d_b;
	    //Device output vector
	    double *d_c;

	    // Size, in bytes, of each vector
	    size_t bytes = n*sizeof(double);

	    // Allocate memory for each vector on host
	    h_a = (double*)malloc(bytes);
	    h_b = (double*)malloc(bytes);
	    h_c = (double*)malloc(bytes);

	    // Allocate memory for each vector on GPU
	    cudaMalloc(&d_a, bytes);
	    cudaMalloc(&d_b, bytes);
	    cudaMalloc(&d_c, bytes);

	    int i;
	    // Initialize vectors on host
	    for( i = 0; i < n; i++ ) {
	        h_a[i] = sin(i)*sin(i);
	        h_b[i] = cos(i)*cos(i);
	    }

	    // Copy host vectors to device
	    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
	    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

	    int blockSize, gridSize;

	    // Number of threads in each thread block
	    blockSize = 1024;

	    // Number of thread blocks in grid
	    gridSize = (int)ceil((float)n/blockSize);

	    // Execute the kernel
	    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

	    // Copy array back to host
	    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

	    // Sum up vector c and print result divided by n, this should equal 1 within error
	    double sum = 0;
	    for(i=0; i<n; i++)
	        sum += h_c[i];
	    printf("final result: %f\n", sum/n);

	    // Release device memory
	    cudaFree(d_a);
	    cudaFree(d_b);
	    cudaFree(d_c);

	    // Release host memory
	    free(h_a);
	    free(h_b);
	    free(h_c);
	return true;
}

void ethash_cuda_miner::hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count)
{

}


void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{

}

