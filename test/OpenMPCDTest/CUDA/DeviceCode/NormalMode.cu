/**
 * @file
 * Tests functionality of `OpenMPCD::CUDA::DeviceCode::NormalMode`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceCode/NormalMode.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/NormalMode.hpp>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/scoped_array.hpp>

#include <vector>


static const double shifts[] = {0.0, -0.5};

#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)


static Catch::Detail::Approx Approx(const float value)
{
        return
                Catch::Detail::Approx(value).
                epsilon(std::numeric_limits<float>::epsilon() * 625);
}
static Catch::Detail::Approx Approx(const double value)
{
        return Catch::Detail::Approx(value);
}

template<typename T>
static __global__ void test_computeNormalCoordinate_1_kernel_setup(
	OpenMPCD::Vector3D<T>* const vectors, const std::size_t totalVectorCount)
{
	for(unsigned int i = 0; i < totalVectorCount; ++i)
		new(vectors + i) OpenMPCD::Vector3D<T>(0.1 * i, -0.33 * i, 0.5 + i);
}
template<typename T>
static __global__ void test_computeNormalCoordinate_1_kernel_teardown(
	OpenMPCD::Vector3D<T>* const vectors, const std::size_t totalVectorCount)
{
	for(unsigned int i = 0; i < totalVectorCount; ++i)
		vectors[i].~Vector3D();
}
template<typename T>
static __global__ void test_computeNormalCoordinate_1_kernel(
	const OpenMPCD::Vector3D<T>* const vectors,
	const unsigned int N,
	const std::size_t i,
	T* const result,
	const T shift)
{
	using OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinate;

	const OpenMPCD::Vector3D<T> r =
		computeNormalCoordinate(i, vectors, N, shift);

	result[0] = r.getX();
	result[1] = r.getY();
	result[2] = r.getZ();
}
template<typename T>
static __global__ void test_computeNormalCoordinate_2_kernel(
	const T* const vectors,
	const unsigned int N,
	const std::size_t i,
	T* const result,
	const T shift)
{
	using OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinate;

	const OpenMPCD::Vector3D<T> r =
		computeNormalCoordinate(i, vectors, N, shift);

	result[0] = r.getX();
	result[1] = r.getY();
	result[2] = r.getZ();
}
template<typename T>
static void test_computeNormalCoordinate(const T shift)
{
	static const unsigned int totalVectorCount = 50;

	typedef OpenMPCD::Vector3D<T> V;

	std::vector<V> vectors;
	std::vector<T> vectors_raw;
	for(unsigned int i = 0; i < totalVectorCount; ++i)
	{
		V v(0.1 * i, -0.33 * i, 0.5 + i);
		vectors.push_back(v);
		vectors_raw.push_back(v.getX());
		vectors_raw.push_back(v.getY());
		vectors_raw.push_back(v.getZ());
	}


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	V* const d_vectors = dmm.allocateMemory<V>(totalVectorCount);
	T* const d_vectors_raw = dmm.allocateMemory<T>(3 * totalVectorCount);
	T* const d_result = dmm.allocateMemory<T>(3);
	T* const d_result_raw = dmm.allocateMemory<T>(3);

	test_computeNormalCoordinate_1_kernel_setup
		<<<1, 1>>>(d_vectors, totalVectorCount);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	dmm.copyElementsFromHostToDevice(
		&vectors_raw[0], d_vectors_raw, 3 * totalVectorCount);

	for(unsigned int N = 1; N <= totalVectorCount; ++N)
	{
		for(unsigned int i = 0; i <= N; ++i)
		{
			test_computeNormalCoordinate_1_kernel<T>
				<<<1, 1>>>(d_vectors, N, i, d_result, shift);
			test_computeNormalCoordinate_2_kernel<T>
				<<<1, 1>>>(d_vectors_raw, N, i, d_result_raw, shift);
			cudaDeviceSynchronize();
			OPENMPCD_CUDA_THROW_ON_ERROR;

			T result[3];
			dmm.copyElementsFromDeviceToHost(d_result, result, 3);

			T result_raw[3];
			dmm.copyElementsFromDeviceToHost(d_result_raw, result_raw, 3);

			const V expected =
				OpenMPCD::NormalMode::computeNormalCoordinate(
					i, &vectors[0], N, shift);

			REQUIRE(expected.getX() == Approx(result[0]));
			REQUIRE(expected.getY() == Approx(result[1]));
			REQUIRE(expected.getZ() == Approx(result[2]));

			for(std::size_t i = 0; i < 3; ++i)
				REQUIRE(result[i] == result_raw[i]);
		}
	}

	test_computeNormalCoordinate_1_kernel_teardown
		<<<1, 1>>>(d_vectors, totalVectorCount);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinate`",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_computeNormalCoordinate<T>(T(shift));

	for(std::size_t i = 0; i < sizeof(shifts)/sizeof(shifts[0]); ++i)
	{
		const double shift = shifts[i];

		BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			SEQ_DATATYPE)
	}

	#undef CALLTEST
}



template<typename T>
static __global__ void test_computeNormalCoordinates_testkernel(
	const T* const vectors,
	const unsigned int N,
	T* const computationResults,
	bool* const testResult,
	const T shift)
{
	using OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinate;
	using OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinates;

	*testResult = false;

	computeNormalCoordinates(vectors, N, computationResults, shift);

	for(std::size_t i = 0; i <= N; ++i)
	{
		const OpenMPCD::Vector3D<T> expected =
			computeNormalCoordinate(i, vectors, N, shift);
		const OpenMPCD::RemotelyStoredVector<T> result(computationResults, i);

		if(result != expected)
			return;
	}

	*testResult = true;
}
template<typename T>
static void test_computeNormalCoordinates(const T shift)
{
	static const unsigned int totalVectorCount = 100;

	typedef OpenMPCD::Vector3D<T> V;

	std::vector<T> vectors_raw;
	for(unsigned int i = 0; i < totalVectorCount; ++i)
	{
		V v(0.1 * i, -0.33 * i, 0.5 + i);
		vectors_raw.push_back(v.getX());
		vectors_raw.push_back(v.getY());
		vectors_raw.push_back(v.getZ());
	}


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* const d_vectors = dmm.allocateMemory<T>(3 * totalVectorCount);
	T* const d_computationResults =
		dmm.allocateMemory<T>(3 * (totalVectorCount + 1));
	bool* const d_testResult = dmm.allocateMemory<bool>(1);

	dmm.copyElementsFromHostToDevice(
		&vectors_raw[0], d_vectors, 3 * totalVectorCount);
	dmm.zeroMemory(d_computationResults, 3 * (totalVectorCount + 1));
	dmm.zeroMemory(d_testResult, 1);

	for(unsigned int N = 1; N <= totalVectorCount; ++N)
	{
		test_computeNormalCoordinates_testkernel<T>
			<<<1, 1>>>(d_vectors, N, d_computationResults, d_testResult, shift);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		bool testResult = false;
		dmm.copyElementsFromDeviceToHost(d_testResult, &testResult, 1);

		REQUIRE(testResult);
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinates`",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_computeNormalCoordinates<T>(T(shift));

	for(std::size_t i = 0; i < sizeof(shifts)/sizeof(shifts[0]); ++i)
	{
		const double shift = shifts[i];

		BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			SEQ_DATATYPE)
	}

	#undef CALLTEST
}



template<typename T>
static void test_computeNormalCoordinates_kernel(const T shift)
{
	static const unsigned int chainLength = 11;
	static const unsigned int chainCount = 17;

	static const unsigned int particleCount = chainCount * chainLength;
	static const unsigned int modeCount = chainLength + 1;
	static const unsigned int resultsSize = 3 * chainCount * modeCount;

	std::vector<OpenMPCD::Vector3D<T> > vectors;
	std::vector<T> vectors_raw;
	for(unsigned int i = 0; i < particleCount; ++i)
	{
		OpenMPCD::Vector3D<T> v(0.1 * i, -0.33 * i, 0.5 + i);
		vectors.push_back(v);
		vectors_raw.push_back(v.getX());
		vectors_raw.push_back(v.getY());
		vectors_raw.push_back(v.getZ());
	}


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* const d_vectors = dmm.allocateMemory<T>(3 * particleCount);
	T* const d_computationResults =	dmm.allocateMemory<T>(resultsSize);

	dmm.copyElementsFromHostToDevice(
		&vectors_raw[0], d_vectors, 3 * particleCount);
	dmm.zeroMemory(d_computationResults, resultsSize);

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN(chainCount, 3, 5)
		OpenMPCD::CUDA::DeviceCode::NormalMode::
		computeNormalCoordinates<T> <<<gridSize, blockSize>>>(
			workUnitOffset,
			chainLength,
			chainCount,
			d_vectors,
			d_computationResults,
			shift);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	boost::scoped_array<T> results(new T[resultsSize]);
	dmm.copyElementsFromDeviceToHost(
		d_computationResults, results.get(), resultsSize);

	for(unsigned int chain = 0; chain < chainCount; ++chain)
	{
		const OpenMPCD::Vector3D<T>* const positions =
			&vectors[chainLength * chain];

		for(unsigned int mode = 0; mode < modeCount; ++mode)
		{
			const OpenMPCD::Vector3D<T> expected =
				OpenMPCD::NormalMode::computeNormalCoordinate(
					mode, positions, chainLength, shift);

			const unsigned int resultsOffset = 3 * (modeCount * chain + mode);

			REQUIRE(expected.getX() == Approx(results[resultsOffset + 0]));
			REQUIRE(expected.getY() == Approx(results[resultsOffset + 1]));
			REQUIRE(expected.getZ() == Approx(results[resultsOffset + 2]));
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::NormalMode::computeNormalCoordinates` kernel",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_computeNormalCoordinates_kernel<T>(T(shift));

	for(std::size_t i = 0; i < sizeof(shifts)/sizeof(shifts[0]); ++i)
	{
		const double shift = shifts[i];

		BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			SEQ_DATATYPE)
	}

	#undef CALLTEST
}
