/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Distributions::Uniform0e1i`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Distributions/Uniform0e1i.hpp>

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
	OpenMPCD::CUDA::Random::Distributions::Uniform0e1i<T> dist;

	for(std::size_t i = 0; i < count; ++i)
		output[i] = dist(rng);
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Uniform0e1i` "
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
__global__ void test_generateValues_2results(
	T* const output,
	const std::size_t count,
	const unsigned long long seed,
	const unsigned long long subsequence)
{
	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, subsequence);
	OpenMPCD::CUDA::Random::Distributions::Uniform0e1i<T> dist;

	OPENMPCD_DEBUG_ASSERT(count % 2 == 0);

	for(std::size_t i = 0; i < count; i += 2)
		dist(rng, &output[i], &output[i + 1]);
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Uniform0e1i` "
		"returns expected values, two-result version",
	"[CUDA]")
{
	static const std::size_t count = 10;
	static const unsigned long long seed = 1337;
	static const unsigned long long subsequence = 42;

	REQUIRE(count % 2 == 0); //assumption in test_generateValues_2results


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
				0.8675456197150472892332118135527707636356353759765625,
				0.7330651376161412269283346176962368190288543701171875,
				0.7815457564396315337518217347678728401660919189453125,
				0.194764188232835533387543591743451543152332305908203125,
				0.009873982033111239342559883880312554538249969482421875,
				0.5097126194990158065678542698151431977748870849609375,
				0.362274460013014942827425102223060093820095062255859375,
				0.998664764317479747290917657664977014064788818359375,
				0.283656109119770627469137025400414131581783294677734375,
				0.285399475967528848396881357984966598451137542724609375
		};


	test_generateValues_2results<<<1, 1>>>(
		d_output_float, count, seed, subsequence);
	test_generateValues_2results<<<1, 1>>>(
		d_output_double, count, seed, subsequence);
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
	OpenMPCD::CUDA::Random::Distributions::Uniform0e1i<T> dist;

	for(std::size_t i = 0; i < count; ++i)
	{
		const T value = dist(rng);
		if(value <= 0 || value > 1)
			*status = false;
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Uniform0e1i` "
		"returns values in the range `(0, 1]`",
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


template<typename T>
static
__global__ void test_checkRange_2results(
	bool* const status,
	const std::size_t count,
	const unsigned long long seed)
{
	*status = true;

	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, 0);
	OpenMPCD::CUDA::Random::Distributions::Uniform0e1i<T> dist;

	for(std::size_t i = 0; i < count; ++i)
	{
		T value1;
		T value2;

		dist(rng, &value1, &value2);

		if(value1 <= 0 || value1 > 1)
			*status = false;
		if(value2 <= 0 || value2 > 1)
			*status = false;
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Uniform0e1i` "
		"returns values in the range `(0, 1], two-result version`",
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

	test_checkRange_2results<float><<<1, 1>>>(d_status_float, count, seed);
	test_checkRange_2results<double><<<1, 1>>>(d_status_double, count, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	static const bool true_ = true;
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, d_status_float, 1));
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, d_status_double, 1));
}
