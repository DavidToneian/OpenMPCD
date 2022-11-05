/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Distributions::Uniform0e1e`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Distributions/Uniform0e1e.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Generators/Philox4x32-10.hpp>

#include <ctime>


template<typename T>
static
__global__ void test_generateValues(
	T* const output,
	const std::size_t count,
	const unsigned long long seed,
	const unsigned long long subsequence)
{
	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, subsequence);
	OpenMPCD::CUDA::Random::Distributions::Uniform0e1e<T> dist;

	for(std::size_t i = 0; i < count; ++i)
		output[i] = dist(rng);
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Uniform0e1e` "
		"returns expected values",
	"[CUDA]")
{
	static const std::size_t count = 10;
	static const unsigned long long seed = 1337;
	static const unsigned long long subsequence = 42;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	float* d_output_float;
	double* d_output_double;
	dmm.allocateMemory(&d_output_float, count);
	dmm.allocateMemory(&d_output_double, count);

	static const float expected_float[count] =
		{
				0.732648551464080810546875,
				0.86754596233367919921875,
				0.872021019458770751953125,
				0.733065545558929443359375,
				0.3170096576213836669921875,
				0.781545817852020263671875,
				0.108345709741115570068359375,
				0.19476413726806640625,
				0.927692115306854248046875,
				0.00987425632774829864501953125
		};
	static const double expected_double[count] =
		{
				0.732648525969125330448150634765625,
				0.867545939167030155658721923828125,
				0.872021028189919888973236083984375,
				0.733065537759102880954742431640625,
				0.317009667516686022281646728515625,
				0.781545844278298318386077880859375,
				0.108345711254514753818511962890625,
				0.194764139014296233654022216796875,
				0.927692140801809728145599365234375,
				0.009874255978502333164215087890625
		};


	test_generateValues<<<1, 1>>>(d_output_float, count, seed, subsequence);
	test_generateValues<<<1, 1>>>(d_output_double, count, seed, subsequence);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expected_float, d_output_float, count));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expected_double, d_output_double, count));
}


template<typename T>
static
__global__ void test_checkRange(
	bool* const status,
	const std::size_t count,
	const unsigned long long seed)
{
	*status = true;

	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, 0);
	OpenMPCD::CUDA::Random::Distributions::Uniform0e1e<T> dist;

	for(std::size_t i = 0; i < count; ++i)
	{
		const T value = dist(rng);
		if(value <= 0 || value >= 1)
			*status = false;
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Uniform0e1e` "
		"returns values in the range `(0, 1)`",
	"[CUDA]")
{
	static const std::size_t count = 1000;

	const unsigned long long seed = std::time(0);

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);
	bool* d_status_float;
	bool* d_status_double;
	dmm.allocateMemory(&d_status_float, 1);
	dmm.allocateMemory(&d_status_double, 1);

	dmm.zeroMemory(d_status_float, 1);
	dmm.zeroMemory(d_status_double, 1);

	test_checkRange<float><<<1, 1>>>(d_status_float, count, seed);
	test_checkRange<double><<<1, 1>>>(d_status_double, count, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	static const bool true_ = true;
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, d_status_float, 1));
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, d_status_double, 1));
}
