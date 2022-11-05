/**
 * @file
 * Tests `OpenMPCD::RemotelyStoredVector`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Types.hpp>

#include <boost/preprocessor/seq/for_each.hpp>

template<typename T>
static __global__ void test_atomicAdd_Vector3D_kernel(T* const target)
{
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	const T incrementX = 1.0 / id;
	const T incrementY = - 0.5 / id;
	const T incrementZ = 12.34 / id;
	const OpenMPCD::Vector3D<T> increment(incrementX, incrementY, incrementZ);

	OpenMPCD::RemotelyStoredVector<T> rsv(target, 0);
	rsv.atomicAdd(increment);
}

template<typename T>
static void test_atomicAdd_Vector3D()
{
	static const std::size_t gridSize = 2;
	static const std::size_t blockSize = 512;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* const target = dmm.allocateMemory<T>(3);
	dmm.zeroMemory(target, 3);

	test_atomicAdd_Vector3D_kernel<T> <<<gridSize, blockSize>>>(target);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	T expected[3] = {0};
	for(std::size_t block = 0; block < gridSize; ++block)
	{
		for(std::size_t thread = 0; thread < blockSize; ++thread)
		{
			const unsigned int id = block * blockSize + thread;

			const T incrementX = 1.0 / id;
			const T incrementY = - 0.5 / id;
			const T incrementZ = 12.34 / id;

			expected[0] += incrementX;
			expected[1] += incrementY;
			expected[2] += incrementZ;
		}
	}

	T result[3];
	dmm.copyElementsFromDeviceToHost(target, result, 3);

	REQUIRE(result[0] == expected[0]);
	REQUIRE(result[1] == expected[1]);
	REQUIRE(result[2] == expected[2]);
}

SCENARIO(
	"`OpenMPCD::RemotelyStoredVector::atomicAdd`, argument `Vector3D`",
	"[CUDA]")
{
	#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_atomicAdd_Vector3D<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}



template<typename T>
static __global__ void test_operatorEquality_Vector3D_kernel(bool* const result)
{
	*result = false;

	T raw[] = {0, 0, 0, -0.5, 0, 1.5, 0, 0, 0};
	const OpenMPCD::RemotelyStoredVector<T> rsv(raw, 1);
	const OpenMPCD::RemotelyStoredVector<const T> c_rsv(raw, 1);

	const OpenMPCD::Vector3D<T> zero(0, 0, 0);
	const OpenMPCD::Vector3D<T> diff1(0.5, 0, 1.5);
	const OpenMPCD::Vector3D<T> diff2(-0.5, 1.5, 1.5);
	const OpenMPCD::Vector3D<T> diff3(-0.5, 0, 1.55);
	const OpenMPCD::Vector3D<T> same(-0.5, 0, 1.5);

	if(rsv == zero || c_rsv == zero)
		return;
	if(rsv == diff1 || c_rsv == diff1)
		return;
	if(rsv == diff2 || c_rsv == diff2)
		return;
	if(rsv == diff3 || c_rsv == diff3)
		return;

	if(!(rsv == same) || !(c_rsv == same))
		return;


	raw[3] = 0.5;

	if(rsv == zero || c_rsv == zero)
		return;
	if(rsv == same || c_rsv == same)
		return;
	if(rsv == diff2 || c_rsv == diff2)
		return;
	if(rsv == diff3 || c_rsv == diff3)
		return;

	if(!(rsv == diff1) || !(c_rsv == diff1))
		return;

	*result = true;
}

template<typename T>
static void test_operatorEquality_Vector3D()
{
	T raw[] = {0, 0, 0, -0.5, 0, 1.5, 0, 0, 0};
	const OpenMPCD::RemotelyStoredVector<T> rsv(raw, 1);
	const OpenMPCD::RemotelyStoredVector<const T> c_rsv(raw, 1);

	const OpenMPCD::Vector3D<T> zero(0, 0, 0);
	const OpenMPCD::Vector3D<T> diff1(0.5, 0, 1.5);
	const OpenMPCD::Vector3D<T> diff2(-0.5, 1.5, 1.5);
	const OpenMPCD::Vector3D<T> diff3(-0.5, 0, 1.55);
	const OpenMPCD::Vector3D<T> same(-0.5, 0, 1.5);

	REQUIRE_FALSE(rsv == zero);
	REQUIRE_FALSE(c_rsv == zero);
	REQUIRE_FALSE(rsv == diff1);
	REQUIRE_FALSE(c_rsv == diff1);
	REQUIRE_FALSE(rsv == diff2);
	REQUIRE_FALSE(c_rsv == diff2);
	REQUIRE_FALSE(rsv == diff3);
	REQUIRE_FALSE(c_rsv == diff3);
	REQUIRE(rsv == same);
	REQUIRE(c_rsv == same);


	raw[3] = 0.5;

	REQUIRE_FALSE(rsv == zero);
	REQUIRE_FALSE(c_rsv == zero);
	REQUIRE(rsv == diff1);
	REQUIRE(c_rsv == diff1);
	REQUIRE_FALSE(rsv == diff2);
	REQUIRE_FALSE(c_rsv == diff2);
	REQUIRE_FALSE(rsv == diff3);
	REQUIRE_FALSE(c_rsv == diff3);
	REQUIRE_FALSE(rsv == same);
	REQUIRE_FALSE(c_rsv == same);



	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* const d_result = dmm.allocateMemory<bool>(1);
	dmm.zeroMemory(d_result, 1);

	test_operatorEquality_Vector3D_kernel<T> <<<1, 1>>>(d_result);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	bool result = false;
	dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
	REQUIRE(result);
}

SCENARIO(
	"`OpenMPCD::RemotelyStoredVector::operator==`, argument `Vector3D`",
	"[CUDA]")
{
	#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_operatorEquality_Vector3D<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}




template<typename T>
static __global__
void test_operatorInequality_Vector3D_kernel(bool* const result)
{
	*result = false;

	T raw[] = {0, 0, 0, -0.5, 0, 1.5, 0, 0, 0};
	const OpenMPCD::RemotelyStoredVector<T> rsv(raw, 1);
	const OpenMPCD::RemotelyStoredVector<const T> c_rsv(raw, 1);

	const OpenMPCD::Vector3D<T> zero(0, 0, 0);
	const OpenMPCD::Vector3D<T> diff1(0.5, 0, 1.5);
	const OpenMPCD::Vector3D<T> diff2(-0.5, 1.5, 1.5);
	const OpenMPCD::Vector3D<T> diff3(-0.5, 0, 1.55);
	const OpenMPCD::Vector3D<T> same(-0.5, 0, 1.5);

	if(!(rsv != zero) || !(c_rsv != zero))
		return;
	if(!(rsv != diff1) || !(c_rsv != diff1))
		return;
	if(!(rsv != diff2) || !(c_rsv != diff2))
		return;
	if(!(rsv != diff3) || !(c_rsv != diff3))
		return;

	if(rsv != same || c_rsv != same)
		return;


	raw[3] = 0.5;

	if(!(rsv != zero) || !(c_rsv != zero))
		return;
	if(!(rsv != same) || !(c_rsv != same))
		return;
	if(!(rsv != diff2) || !(c_rsv != diff2))
		return;
	if(!(rsv != diff3) || !(c_rsv != diff3))
		return;

	if(rsv != diff1 || c_rsv != diff1)
		return;

	*result = true;
}

template<typename T>
static void test_operatorInequality_Vector3D()
{
	T raw[] = {0, 0, 0, -0.5, 0, 1.5, 0, 0, 0};
	const OpenMPCD::RemotelyStoredVector<T> rsv(raw, 1);
	const OpenMPCD::RemotelyStoredVector<const T> c_rsv(raw, 1);

	const OpenMPCD::Vector3D<T> zero(0, 0, 0);
	const OpenMPCD::Vector3D<T> diff1(0.5, 0, 1.5);
	const OpenMPCD::Vector3D<T> diff2(-0.5, 1.5, 1.5);
	const OpenMPCD::Vector3D<T> diff3(-0.5, 0, 1.55);
	const OpenMPCD::Vector3D<T> same(-0.5, 0, 1.5);

	REQUIRE(rsv != zero);
	REQUIRE(c_rsv != zero);
	REQUIRE(rsv != diff1);
	REQUIRE(c_rsv != diff1);
	REQUIRE(rsv != diff2);
	REQUIRE(c_rsv != diff2);
	REQUIRE(rsv != diff3);
	REQUIRE(c_rsv != diff3);
	REQUIRE_FALSE(rsv != same);
	REQUIRE_FALSE(c_rsv != same);


	raw[3] = 0.5;

	REQUIRE(rsv != zero);
	REQUIRE(c_rsv != zero);
	REQUIRE_FALSE(rsv != diff1);
	REQUIRE_FALSE(c_rsv != diff1);
	REQUIRE(rsv != diff2);
	REQUIRE(c_rsv != diff2);
	REQUIRE(rsv != diff3);
	REQUIRE(c_rsv != diff3);
	REQUIRE(rsv != same);
	REQUIRE(c_rsv != same);



	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* const d_result = dmm.allocateMemory<bool>(1);
	dmm.zeroMemory(d_result, 1);

	test_operatorInequality_Vector3D_kernel<T> <<<1, 1>>>(d_result);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	bool result = false;
	dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
	REQUIRE(result);
}

SCENARIO(
	"`OpenMPCD::RemotelyStoredVector::operator!=`, argument `Vector3D`",
	"[CUDA]")
{
	#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_operatorInequality_Vector3D<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}


template<typename T>
static void test_operatorLeftshift_std_ostream()
{
	static const std::size_t count = 3;
	static const T raw[] = {0, 0, 0, -0.5, 0, 1.5, 0, 0, 0};
	REQUIRE(sizeof(raw)/sizeof(raw[0]) == 3 * count);

	for(unsigned int i = 0; i < count; ++i)
	{
		const OpenMPCD::RemotelyStoredVector<T> rsv(raw, i);
		const OpenMPCD::RemotelyStoredVector<const T> c_rsv(raw, i);

		std::stringstream ss;
		std::stringstream c_ss;
		ss << rsv;
		c_ss << c_rsv;

		std::stringstream expected;
		expected << rsv.getX() << " " << rsv.getY() << " " << rsv.getZ();

		REQUIRE(ss.str() == expected.str());
		REQUIRE(c_ss.str() == expected.str());
	}
}
SCENARIO(
	"friend `OpenMPCD::RemotelyStoredVector::operator<<` for `std::ostream`",
	"[CUDA]")
{
	#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_operatorInequality_Vector3D<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}
