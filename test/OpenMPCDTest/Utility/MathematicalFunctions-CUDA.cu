/**
 * @file
 * Tests CUDA functionality in `OpenMPCD::Utility::MathematicalFunctions`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Utility/MathematicalFunctions.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Types.hpp>

#include <boost/preprocessor/seq/for_each.hpp>


#define TEST_DATATYPES (float)(double)(OpenMPCD::FP)


static const double values[] =
{
	-123.456,
	-1.23,
	-1.0,
	-0.5,
	-0.123,
	0.0,
	0.123,
	0.5,
	1.0,
	1.23,
	123.456
};
static const std::size_t valueCount = sizeof(values)/sizeof(values[0]);



template<typename T>
static __global__ void test_acos_kernel(T* const result, const T x)
{
	*result = OpenMPCD::Utility::MathematicalFunctions::acos(x);
}
template<typename T>
static void test_acos()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < -1 || x > 1)
			continue;

		dmm.zeroMemory(d_result, 1);

		test_acos_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const T expected = OpenMPCD::Utility::MathematicalFunctions::acos(x);

		T result;
		dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
		REQUIRE(result == Approx(expected));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::acos`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_acos<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static __global__ void test_cos_kernel(T* const result, const T x)
{
	*result = OpenMPCD::Utility::MathematicalFunctions::cos(x);
}
template<typename T>
static void test_cos()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		dmm.zeroMemory(d_result, 1);

		test_cos_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const T expected = OpenMPCD::Utility::MathematicalFunctions::cos(x);

		T result;
		dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
		REQUIRE(result == Approx(expected));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::cos`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_cos<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static __global__ void test_cospi_kernel(T* const result, const T x)
{
	*result = OpenMPCD::Utility::MathematicalFunctions::cospi(x);
}
template<typename T>
static void test_cospi()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		dmm.zeroMemory(d_result, 1);

		test_cospi_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const T expected = OpenMPCD::Utility::MathematicalFunctions::cospi(x);

		T result;
		dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
		REQUIRE(result == Approx(expected));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::cospi`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_cospi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static __global__ void test_sin_kernel(T* const result, const T x)
{
	*result = OpenMPCD::Utility::MathematicalFunctions::sin(x);
}
template<typename T>
static void test_sin()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		dmm.zeroMemory(d_result, 1);

		test_sin_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const T expected = OpenMPCD::Utility::MathematicalFunctions::sin(x);

		T result;
		dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
		REQUIRE(result == Approx(expected));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sin`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_sin<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static __global__ void test_sinpi_kernel(T* const result, const T x)
{
	*result = OpenMPCD::Utility::MathematicalFunctions::sinpi(x);
}
template<typename T>
static void test_sinpi()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		dmm.zeroMemory(d_result, 1);

		test_sinpi_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const T expected = OpenMPCD::Utility::MathematicalFunctions::sinpi(x);

		T result;
		dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
		REQUIRE(result == Approx(expected));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sinpi`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_sinpi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}


template<typename T>
static __global__ void test_sincos_kernel(T* const result, const T x)
{
	OpenMPCD::Utility::MathematicalFunctions::sincos(x, &result[0], &result[1]);
}
template<typename T>
static void test_sincos()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 2);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		dmm.zeroMemory(d_result, 2);

		test_sincos_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		T expectedSin = 0;
		T expectedCos = 0;
		OpenMPCD::Utility::MathematicalFunctions::sincos(
			x, &expectedSin, &expectedCos);

		T result[2];
		dmm.copyElementsFromDeviceToHost(d_result, result, 2);
		REQUIRE(result[0] == Approx(expectedSin));
		REQUIRE(result[1] == Approx(expectedCos));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sincos`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_sincos<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static __global__ void test_sincospi_kernel(T* const result, const T x)
{
	OpenMPCD::Utility::MathematicalFunctions::sincospi(
		x, &result[0], &result[1]);
}
template<typename T>
static void test_sincospi()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 2);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		dmm.zeroMemory(d_result, 2);

		test_sincospi_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		T expectedSin = 0;
		T expectedCos = 0;
		OpenMPCD::Utility::MathematicalFunctions::sincospi(
			x, &expectedSin, &expectedCos);

		T result[2];
		dmm.copyElementsFromDeviceToHost(d_result, result, 2);
		REQUIRE(result[0] == Approx(expectedSin));
		REQUIRE(result[1] == Approx(expectedCos));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sincospi`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_sincospi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static __global__ void test_sqrt_kernel(T* const result, const T x)
{
	*result = OpenMPCD::Utility::MathematicalFunctions::sqrt(x);
}
template<typename T>
static void test_sqrt()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < 0)
			continue;

		dmm.zeroMemory(d_result, 1);

		test_sqrt_kernel<<<1, 1>>>(d_result, x);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const T expected = OpenMPCD::Utility::MathematicalFunctions::sqrt(x);

		T result;
		dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
		REQUIRE(result == Approx(expected));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sqrt`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_sqrt<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}
