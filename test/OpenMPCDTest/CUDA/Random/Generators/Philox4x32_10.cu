/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Generators::Philox4x32_10`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Generators/Philox4x32-10.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/random/linear_congruential.hpp>


__global__ void test_generateValues(
	unsigned int* const output,
	const std::size_t count,
	const unsigned long long seed,
	const unsigned long long subsequence)
{
	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, subsequence);

	for(std::size_t i = 0; i < count; ++i)
		output[i] = curand(rng.getState());
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Generators::Philox4x32_10`",
	"[CUDA]")
{
	static const std::size_t count = 10;
	static const unsigned long long seed = 1337;
	static const unsigned long long subsequence = 42;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	unsigned int* d_output;
	dmm.allocateMemory(&d_output, count);

	test_generateValues<<<1, 1>>>(d_output, count, seed, subsequence);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	static const unsigned int expected[count] =
		{
				3146701458,
				3726081436,
				3745301797,
				3148492510,
				1361546154,
				3356713841,
				465341286,
				836505607,
				3984407405,
				42409606,
		};

	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(expected, d_output, count));


	test_generateValues<<<1, 1>>>(d_output, count, seed + 1, subsequence);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(expected, d_output, count));

	test_generateValues<<<1, 1>>>(d_output, count, seed, subsequence + 1);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(expected, d_output, count));
}
