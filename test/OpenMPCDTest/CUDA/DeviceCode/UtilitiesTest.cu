/**
 * @file
 * Tests functionality in `OpenMPCD/CUDA/DeviceCode/Utilities.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Types.hpp>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/scoped_array.hpp>
#include <boost/type_traits/is_signed.hpp>

template<typename T>
static __global__ void test_atomicAdd_scalar_kernel(T* const target)
{
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	const T increment = 1.0 / id;

	OpenMPCD::CUDA::DeviceCode::atomicAdd(target, increment);
}

template<typename T>
static void test_atomicAdd_scalar()
{
	static const std::size_t gridSize = 2;
	static const std::size_t blockSize = 512;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* const target = dmm.allocateMemory<T>(1);
	dmm.zeroMemory(target, 1);

	test_atomicAdd_scalar_kernel<T> <<<gridSize, blockSize>>>(target);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	T expected = 0;
	for(std::size_t block = 0; block < gridSize; ++block)
	{
		for(std::size_t thread = 0; thread < blockSize; ++thread)
		{
			const unsigned int id = block * blockSize + thread;

			const T increment = 1.0 / id;

			expected += increment;
		}
	}

	T result;
	dmm.copyElementsFromDeviceToHost(target, &result, 1);

	REQUIRE(result == expected);
}

SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::atomicAdd`, scalar",
	"[CUDA]")
{
	#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_atomicAdd_scalar<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}



static __device__ bool equalIncludingSpecial(const double x, const double y)
{
	if(isnan(x) && isnan(y))
		return true;
	if(isinf(x) && isinf(y))
		return true;
	return x == y;
}
template<typename T>
static __global__ void test_pow_T_double_kernel(bool* const results)
{
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	bool failure = false;
	for(int exponent = -3; exponent <= 3; ++exponent)
	{
		const T b = static_cast<T>(id);
		const double e = static_cast<double>(exponent) / 2;

		const double bd = static_cast<double>(b);

		if(!equalIncludingSpecial(
			OpenMPCD::CUDA::DeviceCode::pow(b, e), ::pow(bd, e)))
		{
			failure = true;
		}

		if(boost::is_signed<T>::value)
		{
			if(!equalIncludingSpecial(
				OpenMPCD::CUDA::DeviceCode::pow(-b, e),
				::pow(-static_cast<double>(b), e)))
			{
				failure = true;
			}
		}
	}
	{
		using OpenMPCD::CUDA::DeviceCode::pow;
		pow(int(1), 1.0);
		pow(1.0, 1.0);
	}

	results[id] = !failure;
}
template<typename T>
static void test_pow_T_double()
{
	static const std::size_t gridSize = 2;
	static const std::size_t blockSize = 512;
	static const std::size_t elementCount = gridSize * blockSize;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* const d_results = dmm.allocateMemory<bool>(elementCount);
	dmm.zeroMemory(d_results, elementCount);

	test_pow_T_double_kernel<T> <<<gridSize, blockSize>>>(d_results);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	boost::scoped_array<bool> results(new bool[elementCount]);
	dmm.copyElementsFromDeviceToHost(d_results, results.get(), elementCount);

	for(std::size_t i=0; i < elementCount; ++i)
	{
		REQUIRE(results[i]);
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::pow`",
	"[CUDA]")
{
	#define SEQ_DATATYPE \
		(bool)\
		(char)(signed char)(unsigned char)\
		(signed short int)(unsigned short int)\
		(signed int)(unsigned int)\
		(signed long int)(unsigned long int)\
		(double)
	#define CALLTEST(_r, _data, T) \
		test_pow_T_double<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}
