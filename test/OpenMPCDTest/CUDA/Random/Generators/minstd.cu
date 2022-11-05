/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Generators::minstd_rand` and
 * `OpenMPCD::CUDA::Random::Generators::minstd_rand0`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Generators/minstd.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/random/linear_congruential.hpp>


template<typename T>
__global__ void test_generateValues_minstd_rand(
	T* const output, const std::size_t count, const T seed)
{
	OpenMPCD::CUDA::Random::Generators::minstd_rand lcg(seed);

	for(std::size_t i = 0; i < count; ++i)
		output[i] = lcg();
}
template<typename T>
__global__ void test_generateValues_minstd_rand0(
	T* const output, const std::size_t count, const T seed)
{
	OpenMPCD::CUDA::Random::Generators::minstd_rand0 lcg(seed);

	for(std::size_t i = 0; i < count; ++i)
		output[i] = lcg();
}

SCENARIO(
	"`OpenMPCD::CUDA::Random::Generators::minstd_rand`",
	"[CUDA]")
{
	typedef unsigned int T;
	static const std::size_t count = 100;
	static const T seed = 1337;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_output;
	dmm.allocateMemory(&d_output, count);

	test_generateValues_minstd_rand<T><<<1, 1>>>(d_output, count, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	T output[count] = {0};
	boost::random::minstd_rand lcg;
	lcg.seed(seed);
	for(std::size_t i = 0; i < count; ++i)
		output[i] = lcg();

	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(output, d_output, count));
}


SCENARIO(
	"`OpenMPCD::CUDA::Random::Generators::minstd_rand0`",
	"[CUDA]")
{
	typedef unsigned int T;
	static const std::size_t count = 100;
	static const T seed = 1337;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_output;
	dmm.allocateMemory(&d_output, count);

	test_generateValues_minstd_rand0<T><<<1, 1>>>(d_output, count, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	T output[count] = {0};
	boost::random::minstd_rand0 lcg;
	lcg.seed(seed);
	for(std::size_t i = 0; i < count; ++i)
		output[i] = lcg();

	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(output, d_output, count));
}
