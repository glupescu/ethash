README for comilation/execution on FEP-CLUSTER, Tesla C2050 & CUDA 6.0/bin/g
Version 0.1

################################
Cryptocurrency explained
################################
Satoshi Nakamoto's development of Bitcoin in 2009 has often been hailed as a radical development in money and currency, being the first example of a digital asset which simultaneously has no backing or "intrinsic value" and no centralized issuer or controller. 
From a technical standpoint, the ledger of a cryptocurrency such as Bitcoin can be thought of as a state transition system, where there is a "state" consisting of the ownership status of all existing bitcoins and a "state transition function" that takes a state and a transaction and outputs a new state which is the result. In a standard banking system, for example, the state is a balance sheet, a transaction is a request to move $X from A to B, and the state transition function reduces the value in A's account by $X and increases the value in B's account by $X. If A's account has less than $X in the first place, the state transition function returns an error.
 
If we had access to a trustworthy centralized service, this system would be trivial to implement; it could simply be coded exactly as described, using a centralized server's hard drive to keep track of the state. Bitcoin's decentralized consensus process requires nodes in the network to continuously attempt to produce packages of transactions called "blocks". The network is intended to produce roughly one block every ten minutes, with each block containing a timestamp, a nonce, a reference to (ie. hash of) the previous block and a list of all of the transactions that have taken place since the previous block. Over time, this creates a persistent, ever-growing, "blockchain" that constantly updates to represent the latest state of the Bitcoin ledger. The state is not encoded in the block in any way; it is purely an abstraction to be remembered by the validating node and can only be (securely) computed for any block by starting from the genesis state and sequentially applying every transaction in every block. 
The one validity condition present in the above list that is not found in other systems is the requirement for "proof of work". The precise condition is that the double-SHA256 hash of every block, treated as a 256-bit number, must be less than a dynamically adjusted target, which as of the time of this writing is approximately 2187. The purpose of this is to make block creation computationally "hard", thereby preventing sybil attackers from remaking the entire blockchain in their favor. In order to compensate miners for this computational work, the miner of every block is entitled to include a transaction giving themselves 25 BTC out of nowhere. Additionally, if any transaction has a higher total denomination in its inputs than in its outputs, the difference also goes to the miner as a "transaction fee". Incidentally, this is also the only mechanism by which BTC are issued; the genesis state contained no coins at all.

Bitcoin mining explained videos
A1. https://www.youtube.com/watch?v=HrQDMy5SwhE
A2. https://www.youtube.com/watch?v=UrrBcaXuaq8
A3. https://www.youtube.com/watch?v=GmOzih6I1zs
Mining in short refers to: 
•	process transactions (verify/secure, prevent double spending …)
•	create/print new money/currency

################################
Ethereum vs Bitcoin explained
################################

B1. https://github.com/ethereum/wiki/wiki/White-Paper
What Ethereum intends to provide is a blockchain with a built-in fully fledged Turing-complete programming language that can be used to create "contracts" that can be used to encode arbitrary state transition functions, allowing users to create any of the systems described above, as well as many others that we have not yet imagined, simply by writing up the logic in a few lines of code.
In Ethereum, the state is made up of objects called "accounts", with each account having a 20-byte address and state transitions being direct transfers of value and information between accounts. "Ether" is the main internal crypto-fuel of Ethereum, and is used to pay transaction fees. In general, there are two types of accounts: externally owned accounts, controlled by private keys, and contract accounts, controlled by their contract code.
	The Ethereum network includes its own built-in currency, ether, which serves the dual purpose of providing a primary liquidity layer to allow for efficient exchange between various types of digital assets and, more importantly, of providing a mechanism for paying transaction fees. For convenience and to avoid future argument (see the current mBTC/uBTC/satoshi debate in Bitcoin), the denominations will be pre-labelled:
•	1: wei
•	1012: szabo
•	1015: finney
•	1018: ether
This should be taken as an expanded version of the concept of "dollars" and "cents" or "BTC" and "satoshi". In the near future, we expect "ether" to be used for ordinary transactions, "finney" for microtransactions and "szabo" and "wei" for technical discussions around fees and protocol implementation; the remaining denominations may become useful later and should not be included in clients at this point.
The Bitcoin mining algorithm works by having miners compute SHA256 on slightly modified versions of the block header millions of times over and over again, until eventually one node comes up with a version whose hash is less than the target (currently around 2192). However, this mining algorithm is vulnerable to two forms of centralization. First, the mining ecosystem has come to be dominated by ASICs (application-specific integrated circuits), computer chips designed for, and therefore thousands of times more efficient at, the specific task of Bitcoin mining. This means that Bitcoin mining is no longer a highly decentralized and egalitarian pursuit, requiring millions of dollars of capital to effectively participate in. Second, most Bitcoin miners do not actually perform block validation locally; instead, they rely on a centralized mining pool to provide the block headers. This problem is arguably worse: as of the time of this writing, the top three mining pools indirectly control roughly 50% of processing power in the Bitcoin network, although this is mitigated by the fact that miners can switch to other mining pools if a pool or coalition attempts a 51% attack.
The current intent at Ethereum is to use a mining algorithm where miners are required to fetch random data from the state, compute some randomly selected transactions from the last N blocks in the blockchain, and return the hash of the result. This has two important benefits. First, Ethereum contracts can include any kind of computation, so an Ethereum ASIC would essentially be an ASIC for general computation - ie. a better CPU. Second, mining requires access to the entire blockchain, forcing miners to store the entire blockchain and at least be capable of verifying every transaction. This removes the need for centralized mining pools; although mining pools can still serve the legitimate role of evening out the randomness of reward distribution, this function can be served equally well by peer-to-peer pools with no central control.

################################
Algorithm ETHASH description
################################
Details can be found here https://github.com/ethereum/wiki/wiki/Ethash. Ethash is the planned PoW algorithm for Ethereum 1.0. It is the latest version of Dagger-Hashimoto.
The general route that the algorithm takes is as follows:
1.	There exists a seed which can be computed for each block by scanning through the block headers up until that point.
 
2.	From the seed, one can compute a 16 MB pseudorandom cache. Light clients store the cache.
 
3.	From the cache, we can generate a 1 GB dataset, with the property that each item in the dataset depends on only a small number of items from the cache. Full clients and miners store the dataset. The dataset grows linearly with time.
4.	Mining involves grabbing random slices of the dataset and hashing them together. Verification can be done with low memory by using the cache to regenerate the specific pieces of the dataset that you need, so you only need to store the cache
 
################################
Instructiuni Ethash OpenCL
################################

1.	Se creaza un director ethereum-wp unde vor fi toate fisierele, se da clone la repo-ul ethash, se face download la cryptopp si se compileaza. La final incarcam modulele necesare (g++ 4.7, *boost, mpi)
[grigore.lupescu@dp-wn04 ~]$ mkdir ethereum-wp
[grigore.lupescu@dp-wn04 ~]$ cd ethereum-wp
[grigore.lupescu@dp-wn04 ethereum-wp]$ git clone https://github.com/ethereum/ethash.git
[grigore.lupescu@dp-wn04 ethereum-wp]$ wget http://www.cryptopp.com/cryptopp562.zip
[grigore.lupescu@dp-wn04 ethereum-wp]$ mkdir cryptopp
[grigore.lupescu@dp-wn04 ethereum-wp]$ mv cryptopp562.zip cryptopp
[grigore.lupescu@dp-wn04 ethereum-wp]$ cd cryptopp/
[grigore.lupescu@dp-wn04 cryptopp]$ unzip cryptopp562.zip
[grigore.lupescu@dp-wn04 cryptopp]$ make -j20
[grigore.lupescu@dp-wn01 ethereum-wp]$ module load compilers/gnu-4.7.0
.0rigore.lupescu@dp-wn01 ethereum-wp]$ module load libraries/openmpi-1.6-gcc-4.7.0
[grigore.lupescu@dp-wn01 ethereum-wp]$ module load libraries/boost-1.41
[grigore.lupescu@dp-wn01 ethereum-wp]$ module load libraries/cuda-6.0
[grigore.lupescu@dp-wn01 ethereum-wp]$ module list
Currently Loaded Modulefiles:
  1) compilers/gnu-4.7.0               3) libraries/boost-1.41
  2) libraries/openmpi-1.6-gcc-4.7.0   4) libraries/cuda-6.0

2.	Se modifica CMakeLists.txt din ethash in felul urmator:
cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_C_COMPILER /opt/tools/compilers/gnu/4.7.0/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/tools/compilers/gnu/4.7.0/bin/g++)

project(ethash)

set(CRYPTOPP_INCLUDE_DIR /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/cryptopp CACHE FILEPATH "" FORCE)
set(CRYPTOPP_ROOT_DIR /export/home/fils/stud/g/grigore.lupescu/ethereum-wp CACHE FILEPATH "" FORCE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(ETHHASH_LIBS ethash)

include_directories(/opt/tools/libraries/cuda-6.0/include/)
include_directories(/export/home/fils/stud/g/grigore.lupescu/ethereum-wp/cryptopp/)
include_directories(/export/home/fils/stud/g/grigore.lupescu/ethereum-wp/

3.	Se genereaza Makefile folosind cmake:
[grigore.lupescu@dp-wn01 ethereum-wp]$ ls
cryptopp  ethash
[grigore.lupescu@dp-wn01 ethereum-wp]$ mkdir build-ethash
[grigore.lupescu@dp-wn01 ethereum-wp]$ cd build-ethash/
[grigore.lupescu@dp-wn01 build-ethash]$ cmake ../ethash/
-- The C compiler identification is GNU 4.7.0
-- The CXX compiler identification is GNU 4.7.0
-- Check for working C compiler: /opt/tools/compilers/gnu/4.7.0/bin/gcc
-- Check for working C compiler: /opt/tools/compilers/gnu/4.7.0/bin/gcc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working CXX compiler: /opt/tools/compilers/gnu/4.7.0/bin/g++
-- Check for working CXX compiler: /opt/tools/compilers/gnu/4.7.0/bin/g++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Found CryptoPP: /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/cryptopp (Required is at least version "5.6.2")
-- Looking for CL_VERSION_2_0
-- Looking for CL_VERSION_2_0 - not found
-- Looking for CL_VERSION_1_2
-- Looking for CL_VERSION_1_2 - not found
-- Looking for CL_VERSION_1_1
-- Looking for CL_VERSION_1_1 - found
-- Found OpenCL: /usr/lib64/libOpenCL.so (found version "1.1")
-- Found MPI_C: /opt/tools/libraries/openmpi/1.6-gcc-4.7.0/lib/libmpi.so;/usr/lib64/librdmacm.so;/usr/lib64/libibverbs.so;/usr/lib64/librt.so;/usr/lib64/libnsl.so;/usr/lib64/libutil.so;/lib64/libm.so;/usr/lib64/libdl.so;/lib64/libm.so;/usr/lib64/librt.so;/usr/lib64/libnsl.so;/usr/lib64/libutil.so;/lib64/libm.so;/usr/lib64/libdl.so
-- Found MPI_CXX: /opt/tools/libraries/openmpi/1.6-gcc-4.7.0/lib/libmpi_cxx.so;/opt/tools/libraries/openmpi/1.6-gcc-4.7.0/lib/libmpi.so;/usr/lib64/librdmacm.so;/usr/lib64/libibverbs.so;/usr/lib64/librt.so;/usr/lib64/libnsl.so;/usr/lib64/libutil.so;/lib64/libm.so;/usr/lib64/libdl.so;/lib64/libm.so;/usr/lib64/librt.so;/usr/lib64/libnsl.so;/usr/lib64/libutil.so;/lib64/libm.so;/usr/lib64/libdl.so
-- Looking for include file pthread.h
-- Looking for include file pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Could NOT find Boost  ?---------------------------- E OK si fara BOOST
-- Configuring done
-- Generating done
-- Build files have been written to: /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/build-ethash
[grigore.lupescu@dp-wn01 build-ethash]$ -- Build files have been written to: /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/build-ethash

4.	Se elimina optiunea de compilare “-Werror” din CMakelists.txt corespunzator lui libethash-cl (/ethereum-wp/ethash/src/libethash-cl/CMakeLists.txt, linia 44) 
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Wextra -pedantic -fPIC ${CMAKE_CXX_FLAGS}")

5.	Se face compilarea “ethash” si “Benchmark_CL”
[grigore.lupescu@dp-wn01 build-ethash]$ make
Scanning dependencies of target ethash
[ 16%] Building C object src/libethash/CMakeFiles/ethash.dir/util.c.o
[ 33%] Building C object src/libethash/CMakeFiles/ethash.dir/io.c.o
[ 50%] Building C object src/libethash/CMakeFiles/ethash.dir/internal.c.o
[ 66%] Building C object src/libethash/CMakeFiles/ethash.dir/io_posix.c.o
[ 83%] Building CXX object src/libethash/CMakeFiles/ethash.dir/sha3_cryptopp.cpp.o
Linking CXX static library libethash.a
[ 83%] Built target ethash
Scanning dependencies of target ethash-cl
[100%] Building CXX object src/libethash-cl/CMakeFiles/ethash-cl.dir/ethash_cl_miner.cpp.o
In file included from /opt/tools/libraries/cuda-6.0/include/CL/opencl.h:44:0,
                 from /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/ethash/src/libethash-cl/cl.hpp:170,
                 from /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/ethash/src/libethash-cl/ethash_cl_miner.h:5,
                 from /export/home/fils/stud/g/grigore.lupescu/ethereum-wp/ethash/src/libethash-cl/ethash_cl_miner.cpp:30:
/opt/tools/libraries/cuda-6.0/include/CL/cl_gl_ext.h:44:4: warning: "/*" within comment [-Wcomment]
Linking CXX static library libethash-cl.a
[100%] Built target ethash-cl
[grigore.lupescu@dp-wn01 build-ethash]$ make Benchmark_CL
[ 71%] Built target ethash
[ 85%] Built target ethash-cl
Scanning dependencies of target Benchmark_CL
[100%] Building CXX object src/benchmark/CMakeFiles/Benchmark_CL.dir/benchmark.cpp.o
Linking CXX executable Benchmark_CL
[100%] Built target Benchmark_CL

6.	Se ruleaza “./Benchmark_CL”
[grigore.lupescu@dp-wn02 benchmark]$ ./Benchmark_CL
ethash_mkcache: 0ms, sha3: 6a286c5fc0f36814732c86c3e71c036dd96d58def86b9244bb1480571e67d2a8
ethash_light test: 2ms, a7ea1de3a8007134900cd2c86f7e55af68a1d3e4537438a0a966b6cbafa23c90
Using platform: NVIDIA CUDA
Using device: Tesla M2070 (OpenCL 1.1 CUDA)
ethash_cl_miner init: 242363ms
found: 00000000006a9aa9 -> 93456708ba3d2bd90ca82141ee0d95570a958baf90091cd396f51fa374000000
found: 00000000015767f5 -> ec52f770e3d1667d5da6673f2fa07417076c2a6de4f822b4281a81768b000000
Search took: 5050ms
hashrate:     6.64 Mh/s, bw:    50.69 GB/s





