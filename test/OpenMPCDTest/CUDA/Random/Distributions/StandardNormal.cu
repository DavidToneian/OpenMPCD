/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Distributions::StandardNormal`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Distributions/StandardNormal.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Generators/Philox4x32-10.hpp>

#include <ctime>

#include <iostream>
#include <fstream>

template<typename T>
static
__global__ void test_generateValues(
	T* const output,
	const std::size_t count,
	const unsigned long long seed,
	const unsigned long long subsequence)
{
	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, subsequence);
	OpenMPCD::CUDA::Random::Distributions::StandardNormal<T> dist;

	for(std::size_t i = 0; i < count; ++i)
		output[i] = dist(rng);
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::StandardNormal` "
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
			-0.58325493335723876953125,
			0.53102886676788330078125,
			-0.520379006862640380859375,
			-0.0555795095860958099365234375,
			-1.4861209392547607421875,
			0.298480093479156494140625,
			1.98258221149444580078125,
			0.71709430217742919921875,
			0.0240220837295055389404296875,
			0.3866957128047943115234375
		};
	static const double expected_double[count] =
		{
			-0.53006515125737563298713439507992006838321685791015625,
			-0.05661534575131328683728071382574853487312793731689453125,
			0.66025184378453538602826711212401278316974639892578125,
			0.238810940776321256606706810998730361461639404296875,
			-0.1853453305948347118459196281037293374538421630859375,
			-3.033372912444662450326404723455198109149932861328125,
			-0.0119551840799749335697566010594528052024543285369873046875,
			1.42497839589412844674143343581818044185638427734375,
			1.5483412554770883406973780438420362770557403564453125,
			-0.35017805815747760078693318064324557781219482421875
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
