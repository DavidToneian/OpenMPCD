/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Generators::LinearCongruent`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Generators/LinearCongruent.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/random/linear_congruential.hpp>


template<typename T, T a, T c, T m>
__global__ void test_generateValues(
	T* const output, const std::size_t count, const T seed)
{
	OpenMPCD::CUDA::Random::Generators::LinearCongruent<T, a, c, m> lcg(seed);

	for(std::size_t i = 0; i < count; ++i)
		output[i] = lcg();
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Generators::LinearCongruent`",
	"[CUDA]")
{
	typedef unsigned int T;
	static const std::size_t count = 100;
	static const T m = 9012;
	static const T a = 1234;
	static const T c = 5678;
	static const T seed = 1337;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_output;
	dmm.allocateMemory(&d_output, count);

	test_generateValues<T, a, c, m><<<1, 1>>>(d_output, count, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	T output[count] = {0};
	boost::random::linear_congruential_engine<T, a, c, m> lcg;
	lcg.seed(seed);
	for(std::size_t i = 0; i < count; ++i)
		output[i] = lcg();

	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(output, d_output, count));
}


SCENARIO(
	"`OpenMPCD::CUDA::Random::Generators::LinearCongruent::LinearCongruent`",
	"")
{
	typedef int T;
	static const T m = 9012;
	static const T a = 1234;
	static const T c = 5678;

	typedef OpenMPCD::CUDA::Random::Generators::LinearCongruent<T, a, c, m> LCG;

	REQUIRE_THROWS_AS((LCG(-1)), OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS((LCG(m)), OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS((LCG(m + 1)), OpenMPCD::InvalidArgumentException);
}
