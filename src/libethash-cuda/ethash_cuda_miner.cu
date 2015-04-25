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

// TODO
#define DAG_SIZE 64
#define MAX_OUTPUTS 64
#define GROUP_SIZE 64
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////




// author Tim Hughes <tim@twistedfury.com>
// Tested on Radeon HD 7850
// Hashrate: 15940347 hashes/s
// Bandwidth: 124533 MB/s
// search kernel should fit in <= 84 VGPRS (3 wavefronts)

#define THREADS_PER_HASH (128 / 16)
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

#define FNV_PRIME	0x01000193

const uint2 Keccak_f1600_RC[24] = {
	{0x00000001, 0x00000000},
	{0x00008082, 0x00000000},
	{0x0000808a, 0x80000000},
	{0x80008000, 0x80000000},
	{0x0000808b, 0x00000000},
	{0x80000001, 0x00000000},
	{0x80008081, 0x80000000},
	{0x00008009, 0x80000000},
	{0x0000008a, 0x00000000},
	{0x00000088, 0x00000000},
	{0x80008009, 0x00000000},
	{0x8000000a, 0x00000000},
	{0x8000808b, 0x00000000},
	{0x0000008b, 0x80000000},
	{0x00008089, 0x80000000},
	{0x00008003, 0x80000000},
	{0x00008002, 0x80000000},
	{0x00000080, 0x80000000},
	{0x0000800a, 0x00000000},
	{0x8000000a, 0x80000000},
	{0x80008081, 0x80000000},
	{0x00008080, 0x80000000},
	{0x80000001, 0x00000000},
	{0x80008008, 0x80000000},
};

__device__ void keccak_f1600_round(uint2* a, uint r, uint out_size)
{
//   #if !__ENDIAN_LITTLE__
//	for (uint i = 0; i != 25; ++i)
//		a[i] = a[i].yx;
//   #endif

	uint2 b[25];
	uint2 t;

	// Theta
	b[0].x = a[0].x ^ a[5].x ^ a[10].x ^ a[15].x ^ a[20].x;
	b[1].x = a[1].x ^ a[6].x ^ a[11].x ^ a[16].x ^ a[21].x;
	b[2].x = a[2].x ^ a[7].x ^ a[12].x ^ a[17].x ^ a[22].x;
	b[3].x = a[3].x ^ a[8].x ^ a[13].x ^ a[18].x ^ a[23].x;
	b[4].x = a[4].x ^ a[9].x ^ a[14].x ^ a[19].x ^ a[24].x;

	b[0].y = a[0].y ^ a[5].y ^ a[10].y ^ a[15].y ^ a[20].y;
	b[1].y = a[1].y ^ a[6].y ^ a[11].y ^ a[16].y ^ a[21].y;
	b[2].y = a[2].y ^ a[7].y ^ a[12].y ^ a[17].y ^ a[22].y;
	b[3].y = a[3].y ^ a[8].y ^ a[13].y ^ a[18].y ^ a[23].y;
	b[4].y = a[4].y ^ a[9].y ^ a[14].y ^ a[19].y ^ a[24].y;

	t.x = b[4].x ^ (b[1].x << 1 | b[1].y >> 31);
	t.y = b[1].y << 1 | b[1].x >> 31;

	a[0].x ^= t.x;
	a[5].x ^= t.x;
	a[10].x ^= t.x;
	a[15].x ^= t.x;
	a[20].x ^= t.x;
	a[0].y ^= t.y;
	a[5].y ^= t.y;
	a[10].y ^= t.y;
	a[15].y ^= t.y;
	a[20].y ^= t.y;
/*
	t = b[0] ^ (uint2)(b[2].x << 1 | b[2].y >> 31, b[2].y << 1 | b[2].x >> 31);
	a[1] ^= t;
	a[6] ^= t;
	a[11] ^= t;
	a[16] ^= t;
	a[21] ^= t;
	t = b[1] ^ (uint2)(b[3].x << 1 | b[3].y >> 31, b[3].y << 1 | b[3].x >> 31);
	a[2] ^= t;
	a[7] ^= t;
	a[12] ^= t;
	a[17] ^= t;
	a[22] ^= t;
	t = b[2] ^ (uint2)(b[4].x << 1 | b[4].y >> 31, b[4].y << 1 | b[4].x >> 31);
	a[3] ^= t;
	a[8] ^= t;
	a[13] ^= t;
	a[18] ^= t;
	a[23] ^= t;
	t = b[3] ^ (uint2)(b[0].x << 1 | b[0].y >> 31, b[0].y << 1 | b[0].x >> 31);
	a[4] ^= t;
	a[9] ^= t;
	a[14] ^= t;
	a[19] ^= t;
	a[24] ^= t;

	// Rho Pi
	b[0] = a[0];
	b[10] = (uint2)(a[1].x << 1 | a[1].y >> 31, a[1].y << 1 | a[1].x >> 31);
	b[7] = (uint2)(a[10].x << 3 | a[10].y >> 29, a[10].y << 3 | a[10].x >> 29);
	b[11] = (uint2)(a[7].x << 6 | a[7].y >> 26, a[7].y << 6 | a[7].x >> 26);
	b[17] = (uint2)(a[11].x << 10 | a[11].y >> 22, a[11].y << 10 | a[11].x >> 22);
	b[18] = (uint2)(a[17].x << 15 | a[17].y >> 17, a[17].y << 15 | a[17].x >> 17);
	b[3] = (uint2)(a[18].x << 21 | a[18].y >> 11, a[18].y << 21 | a[18].x >> 11);
	b[5] = (uint2)(a[3].x << 28 | a[3].y >> 4, a[3].y << 28 | a[3].x >> 4);
	b[16] = (uint2)(a[5].y << 4 | a[5].x >> 28, a[5].x << 4 | a[5].y >> 28);
	b[8] = (uint2)(a[16].y << 13 | a[16].x >> 19, a[16].x << 13 | a[16].y >> 19);
	b[21] = (uint2)(a[8].y << 23 | a[8].x >> 9, a[8].x << 23 | a[8].y >> 9);
	b[24] = (uint2)(a[21].x << 2 | a[21].y >> 30, a[21].y << 2 | a[21].x >> 30);
	b[4] = (uint2)(a[24].x << 14 | a[24].y >> 18, a[24].y << 14 | a[24].x >> 18);
	b[15] = (uint2)(a[4].x << 27 | a[4].y >> 5, a[4].y << 27 | a[4].x >> 5);
	b[23] = (uint2)(a[15].y << 9 | a[15].x >> 23, a[15].x << 9 | a[15].y >> 23);
	b[19] = (uint2)(a[23].y << 24 | a[23].x >> 8, a[23].x << 24 | a[23].y >> 8);
	b[13] = (uint2)(a[19].x << 8 | a[19].y >> 24, a[19].y << 8 | a[19].x >> 24);
	b[12] = (uint2)(a[13].x << 25 | a[13].y >> 7, a[13].y << 25 | a[13].x >> 7);
	b[2] = (uint2)(a[12].y << 11 | a[12].x >> 21, a[12].x << 11 | a[12].y >> 21);
	b[20] = (uint2)(a[2].y << 30 | a[2].x >> 2, a[2].x << 30 | a[2].y >> 2);
	b[14] = (uint2)(a[20].x << 18 | a[20].y >> 14, a[20].y << 18 | a[20].x >> 14);
	b[22] = (uint2)(a[14].y << 7 | a[14].x >> 25, a[14].x << 7 | a[14].y >> 25);
	b[9] = (uint2)(a[22].y << 29 | a[22].x >> 3, a[22].x << 29 | a[22].y >> 3);
	b[6] = (uint2)(a[9].x << 20 | a[9].y >> 12, a[9].y << 20 | a[9].x >> 12);
	b[1] = (uint2)(a[6].y << 12 | a[6].x >> 20, a[6].x << 12 | a[6].y >> 20);

	// Chi
	a[0] = bitselect(b[0] ^ b[2], b[0], b[1]);
	a[1] = bitselect(b[1] ^ b[3], b[1], b[2]);
	a[2] = bitselect(b[2] ^ b[4], b[2], b[3]);
	a[3] = bitselect(b[3] ^ b[0], b[3], b[4]);
	if (out_size >= 4)
	{
		a[4] = bitselect(b[4] ^ b[1], b[4], b[0]);
		a[5] = bitselect(b[5] ^ b[7], b[5], b[6]);
		a[6] = bitselect(b[6] ^ b[8], b[6], b[7]);
		a[7] = bitselect(b[7] ^ b[9], b[7], b[8]);
		a[8] = bitselect(b[8] ^ b[5], b[8], b[9]);
		if (out_size >= 8)
		{
			a[9] = bitselect(b[9] ^ b[6], b[9], b[5]);
			a[10] = bitselect(b[10] ^ b[12], b[10], b[11]);
			a[11] = bitselect(b[11] ^ b[13], b[11], b[12]);
			a[12] = bitselect(b[12] ^ b[14], b[12], b[13]);
			a[13] = bitselect(b[13] ^ b[10], b[13], b[14]);
			a[14] = bitselect(b[14] ^ b[11], b[14], b[10]);
			a[15] = bitselect(b[15] ^ b[17], b[15], b[16]);
			a[16] = bitselect(b[16] ^ b[18], b[16], b[17]);
			a[17] = bitselect(b[17] ^ b[19], b[17], b[18]);
			a[18] = bitselect(b[18] ^ b[15], b[18], b[19]);
			a[19] = bitselect(b[19] ^ b[16], b[19], b[15]);
			a[20] = bitselect(b[20] ^ b[22], b[20], b[21]);
			a[21] = bitselect(b[21] ^ b[23], b[21], b[22]);
			a[22] = bitselect(b[22] ^ b[24], b[22], b[23]);
			a[23] = bitselect(b[23] ^ b[20], b[23], b[24]);
			a[24] = bitselect(b[24] ^ b[21], b[24], b[20]);
		}
	}

	// Iota
	a[0] ^= Keccak_f1600_RC[r];

//   #if !__ENDIAN_LITTLE__
//	for (uint i = 0; i != 25; ++i)
//		a[i] = a[i].yx;
//   #endif
*/
}

__device__ void keccak_f1600_no_absorb(ulong* a, uint in_size, uint out_size, uint isolate)
{
	for (uint i = in_size; i != 25; ++i)
	{
		a[i] = 0;
	}
#if __ENDIAN_LITTLE__
	a[in_size] ^= 0x0000000000000001;
	a[24-out_size*2] ^= 0x8000000000000000;
#else
	a[in_size] ^= 0x0100000000000000;
	a[24-out_size*2] ^= 0x0000000000000080;
#endif

	// Originally I unrolled the first and last rounds to interface
	// better with surrounding code, however I haven't done this
	// without causing the AMD compiler to blow up the VGPR usage.
	uint r = 0;
	do
	{
		// This dynamic branch stops the AMD compiler unrolling the loop
		// and additionally saves about 33% of the VGPRs, enough to gain another
		// wavefront. Ideally we'd get 4 in flight, but 3 is the best I can
		// massage out of the compiler. It doesn't really seem to matter how
		// much we try and help the compiler save VGPRs because it seems to throw
		// that information away, hence the implementation of keccak here
		// doesn't bother.
		if (isolate)
		{
			keccak_f1600_round((uint2*)a, r++, 25);
		}
	}
	while (r < 23);

	// final round optimised for digest size
	keccak_f1600_round((uint2*)a, r++, out_size);
}

#define copy(dst, src, count) for (uint i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

__device__ uint fnv(uint x, uint y)
{
	return x * FNV_PRIME ^ y;
}

__device__ uint4 fnv4(uint4 x, uint4 y)
{
	// TODO
	//return x * FNV_PRIME ^ y;
	return {0, 0, 0, 0};
}

__device__ uint fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

typedef union
{
	ulong ulongs[32 / sizeof(ulong)];
	uint uints[32 / sizeof(uint)];
} hash32_t;

typedef union
{
	ulong ulongs[64 / sizeof(ulong)];
	uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
	uint uints[128 / sizeof(uint)];
	uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

__device__ hash64_t init_hash(const hash32_t const* header, ulong nonce, uint isolate)
{
	hash64_t init;
	uint const init_size = countof(init.ulongs);
	uint const hash_size = countof(header->ulongs);

	// sha3_512(header .. nonce)
	ulong state[25];
	copy(state, header->ulongs, hash_size);
	state[hash_size] = nonce;
	keccak_f1600_no_absorb(state, hash_size + 1, init_size, isolate);

	copy(init.ulongs, state, init_size);
	return init;
}

__device__ uint inner_loop(uint4 init, uint thread_id, uint* share, hash128_t const* g_dag, uint isolate)
{
	uint4 mix = init;

	// share init0
	if (thread_id == 0)
		*share = mix.x;
	//barrier(CLK_LOCAL_MEM_FENCE);
	uint init0 = *share;

	uint a = 0;
	do
	{
		bool update_share = thread_id == (a/4) % THREADS_PER_HASH;

		#pragma unroll
		for (uint i = 0; i != 4; ++i)
		{
			if (update_share)
			{
				uint m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a+i), m[i]) % DAG_SIZE;
			}
			//barrier(CLK_LOCAL_MEM_FENCE);

			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
		}
	}
	while ((a += 4) != (ACCESSES & isolate));

	return fnv_reduce(mix);
}

__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix, uint isolate)
{
	ulong state[25];

	hash32_t hash;
	uint const hash_size = countof(hash.ulongs);
	uint const init_size = countof(init->ulongs);
	uint const mix_size = countof(mix->ulongs);

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->ulongs, init_size);
	copy(state + init_size, mix->ulongs, mix_size);
	keccak_f1600_no_absorb(state, init_size+mix_size, hash_size, isolate);

	// copy out
	copy(hash.ulongs, state, hash_size);
	return hash;
}

__device__ hash32_t compute_hash_simple(
	const hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong nonce,
	uint isolate
	)
{
	hash64_t init = init_hash(g_header, nonce, isolate);

	hash128_t mix;
	for (uint i = 0; i != countof(mix.uint4s); ++i)
	{
		mix.uint4s[i] = init.uint4s[i % countof(init.uint4s)];
	}

	uint mix_val = mix.uints[0];
	uint init0 = mix.uints[0];
	uint a = 0;
	do
	{
		uint pi = fnv(init0 ^ a, mix_val) % DAG_SIZE;
		uint n = (a+1) % countof(mix.uints);

		#pragma unroll
		for (uint i = 0; i != countof(mix.uints); ++i)
		{
			mix.uints[i] = fnv(mix.uints[i], g_dag[pi].uints[i]);
			mix_val = i == n ? mix.uints[i] : mix_val;
		}
	}
	while (++a != (ACCESSES & isolate));

	// reduce to output
	hash32_t fnv_mix;
	for (uint i = 0; i != countof(fnv_mix.uints); ++i)
	{
		fnv_mix.uints[i] = fnv_reduce(mix.uint4s[i]);
	}

	return final_hash(&init, &fnv_mix, isolate);
}

typedef union
{
	struct
	{
		hash64_t init;
		uint pad; // avoid lds bank conflicts
	};
	hash32_t mix;
} compute_hash_share;

__device__ hash32_t compute_hash(
	compute_hash_share* share,
	const hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong nonce,
	uint isolate
	)
{
	uint const gid = 4;//get_global_id(0);

	// Compute one init hash per work item.
	hash64_t init = init_hash(g_header, nonce, isolate);

	// Threads work together in this phase in groups of 8.
	uint const thread_id = gid % THREADS_PER_HASH;
	uint const hash_id = (gid % GROUP_SIZE) / THREADS_PER_HASH;

	hash32_t mix;
	uint i = 0;
	do
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;
		//barrier(CLK_LOCAL_MEM_FENCE);

		uint4 thread_init = share[hash_id].init.uint4s[thread_id % (64 / sizeof(uint4))];
		//barrier(CLK_LOCAL_MEM_FENCE);

		uint thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uints, g_dag, isolate);

		share[hash_id].mix.uints[thread_id] = thread_mix;
		//barrier(CLK_LOCAL_MEM_FENCE);

		if (i == thread_id)
			mix = share[hash_id].mix;
		//barrier(CLK_LOCAL_MEM_FENCE);
	}
	while (++i != (THREADS_PER_HASH & isolate));

	return final_hash(&init, &mix, isolate);
}

__global__ void ethash_hash_simple(
	hash32_t* g_hashes,
	const hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
	)
{
	uint const gid = 4; //get_global_id(0);
	g_hashes[gid] = compute_hash_simple(g_header, g_dag, start_nonce + gid, isolate);
}

__global__ void ethash_search_simple(
	volatile uint* restrict g_output,
	const hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
	)
{
	uint const gid = 0; //get_global_id(0);
	hash32_t hash = compute_hash_simple(g_header, g_dag, start_nonce + gid, isolate);

	if (hash.ulongs[countof(hash.ulongs)-1] < target)
	{
		// TODO
		//uint slot = min(MAX_OUTPUTS, atomic_inc(&g_output[0]) + 1);
		uint slot = 0;
		g_output[slot] = gid;
	}
}

__global__ void ethash_hash(
	hash32_t* g_hashes,
	const hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
	)
{
	compute_hash_share share[HASHES_PER_LOOP];

	uint const gid = 4; //get_global_id(0);
	g_hashes[gid] = compute_hash(share, g_header, g_dag, start_nonce + gid, isolate);
}

__global__ void ethash_search(
	volatile uint* restrict g_output,
	const hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
	)
{
	compute_hash_share share[HASHES_PER_LOOP];

	uint const gid = 4; //get_global_id(0);
	hash32_t hash = compute_hash(share, g_header, g_dag, start_nonce + gid, isolate);

	if (hash.ulongs[countof(hash.ulongs)-1] < target)
	{
		// TODO
		//uint slot = min(MAX_OUTPUTS, atomic_inc(&g_output[0]) + 1);
		uint slot = 0;
		g_output[slot] = gid;
	}
}



/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


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
	char buf[256];
	sprintf(buf, "#define %s %uu\n", id, value);
	source.insert(source.begin(), buf, buf + strlen(buf));
}

ethash_cuda_miner::ethash_cuda_miner()
{
}

void ethash_cuda_miner::finish()
{
}

bool ethash_cuda_miner::init(ethash_params const& params, ethash_h256_t const *seed, unsigned workgroup_size)
{
	// store params
	m_params = params;

	m_params.full_size = m_params.full_size / 1024;

	printf("SIZE = %d\n", m_params.full_size);

	// create buffer for dag
	cudaMalloc(&m_dag, m_params.full_size);

	// create buffer for header
	cudaMalloc(&m_header, 32);

	// compute dag on CPU
	{
		void* cache_mem = malloc(m_params.cache_size + 63);
		ethash_cache cache;
		cache.mem = (void*)(((uintptr_t)cache_mem + 63) & ~63);
		ethash_mkcache(&cache, &m_params, seed);

		// if this throws then it's because we probably need to subdivide the dag uploads for compatibility
		char* dag_ptr = (char*) malloc(m_params.full_size);
		cudaMemcpy( m_dag, dag_ptr, m_params.full_size, cudaMemcpyHostToDevice);
		ethash_compute_full_data(dag_ptr, &m_params, &cache);
		cudaMemcpy( dag_ptr, m_dag, m_params.full_size, cudaMemcpyDeviceToHost );

		delete[] dag_ptr;

		free(cache_mem);
	}

	// create mining buffers
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		cudaMalloc(&m_hash_buf[i], 32*c_hash_batch_size);
		cudaMalloc(&m_search_buf[i], (c_max_search_results + 1) * sizeof(uint32_t));
	}
	return true;
}


struct pending_batch
{
	unsigned base;
	unsigned count;
	unsigned buf;
};

void ethash_cuda_miner::hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count)
{

	std::queue<pending_batch> pending;

	cudaMemcpy( m_header, header, 32, cudaMemcpyHostToDevice);

	unsigned buf = 0;
	for (unsigned i = 0; i < count || !pending.empty(); )
	{
		// how many this batch
		if (i < count)
		{
			unsigned const this_count = std::min(count - i, c_hash_batch_size);
			unsigned const batch_count = std::max(this_count, m_workgroup_size);


			// execute it!

			//pending.push({i, this_count, buf});
			i += this_count;
			buf = (buf + 1) % c_num_buffers;
		}

		// read results
		if (i == count || pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead, but this path isn't that important.
			uint8_t* hashes = (uint8_t*)malloc(batch.count * ETHASH_BYTES* sizeof(uint8_t));
			cudaMemcpy( hashes, m_hash_buf[batch.buf], batch.count * ETHASH_BYTES, cudaMemcpyDeviceToHost );
			memcpy(ret + batch.base*ETHASH_BYTES, hashes, batch.count*ETHASH_BYTES);
			cudaMemcpy( m_hash_buf[batch.buf], hashes, batch.count*ETHASH_BYTES, cudaMemcpyDeviceToHost );
			pending.pop();
		}
	}
}


void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{

}

