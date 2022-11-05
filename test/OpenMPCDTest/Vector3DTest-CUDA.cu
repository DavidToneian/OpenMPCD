/**
 * @file
 * Tests `OpenMPCD::Vector3D` on CUDA Devices.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Vector3D.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Types.hpp>

#include <boost/preprocessor/seq/for_each.hpp>

#define TEST_DATATYPES (float)(double)(OpenMPCD::FP)


template<typename T>
static __global__ void test_getRandomUnitVector_kernel(bool* const result)
{
	static const std::size_t count = 100;

	*result = true;

	OpenMPCD::Vector3D<T>* vectors[count];

	OpenMPCD::CUDA::GPURNG rng(1234, 0);

	for(std::size_t i = 0; i < count; ++i)
	{
		const OpenMPCD::Vector3D<T> random =
			OpenMPCD::Vector3D<T>::getRandomUnitVector(rng);

		if(abs(random.getMagnitudeSquared() - 1) > 1e-6)
			*result = false;

		vectors[i] = new OpenMPCD::Vector3D<T>(random);

		for(std::size_t j = 0; j < i; ++j)
		{
			if(*vectors[j] == random)
				*result = false;
		}
	}

	for(std::size_t i = 0; i < count; ++i)
		delete vectors[i];
}
template<typename T> static void test_getRandomUnitVector()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* d_result;
	dmm.allocateMemory(&d_result, 1);
	dmm.zeroMemory(d_result, 1);

	test_getRandomUnitVector_kernel<T><<<1, 1>>>(d_result);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const bool true_ = true;
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, d_result, 1));
}
SCENARIO(
	"`OpenMPCD::Vector3D::getRandomUnitVector`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_getRandomUnitVector<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}


template<typename T>
static __global__ void test_getUnitVectorFromRandom01_kernel(
	T* const result, const T X_1, const T X_2)
{
	const OpenMPCD::Vector3D<T> v =
		OpenMPCD::Vector3D<T>::getUnitVectorFromRandom01(X_1, X_2);

	result[0] = v.getX();
	result[1] = v.getY();
	result[2] = v.getZ();
}
template<typename T> static void test_getUnitVectorFromRandom01()
{
	static const T X_1 = 0.01234;
	static const T X_2 = 0.56789;


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 3);
	dmm.zeroMemory(d_result, 2);

	test_getUnitVectorFromRandom01_kernel<<<1, 1>>>(d_result, X_1, X_2);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const OpenMPCD::Vector3D<T> expected =
		OpenMPCD::Vector3D<T>::getUnitVectorFromRandom01(X_1, X_2);

	T result[3];
	dmm.copyElementsFromDeviceToHost(d_result, result, 3);
	REQUIRE(result[0] == Approx(expected.getX()));
	REQUIRE(result[1] == Approx(expected.getY()));
	REQUIRE(result[2] == Approx(expected.getZ()));
}
SCENARIO(
	"`OpenMPCD::Vector3D::getUnitVectorFromRandom01`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_getUnitVectorFromRandom01<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}
