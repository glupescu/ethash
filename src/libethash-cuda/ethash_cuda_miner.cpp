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

#define ETHASH_BYTES 32

#undef min
#undef max

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
	return true;
}

void ethash_cuda_miner::hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count)
{

}


void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{

}

