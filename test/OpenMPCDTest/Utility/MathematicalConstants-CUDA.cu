/**
 * @file
 * Tests CUDA functionality in `OpenMPCD::Utility::MathematicalConstants`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Utility/MathematicalConstants.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Types.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/preprocessor/seq/for_each.hpp>


#define TEST_DATATYPES (float)(double)(OpenMPCD::FP)

template<typename T>
static __global__ void test_pi_kernel(T* const result)
{
	*result = OpenMPCD::Utility::MathematicalConstants::pi<T>();
}
template<typename T>
static void test_pi()
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_result;
	dmm.allocateMemory(&d_result, 1);
	dmm.zeroMemory(d_result, 1);

	test_pi_kernel<<<1, 1>>>(d_result);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const T expected = OpenMPCD::Utility::MathematicalConstants::pi<T>();
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&expected, d_result, 1));
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalConstants::pi`, CUDA",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_pi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}
