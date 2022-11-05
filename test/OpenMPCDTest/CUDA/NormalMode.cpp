/**
 * @file
 * Tests functionality of `OpenMPCD::CUDA::NormalMode`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/NormalMode.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/NormalMode.hpp>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/scoped_array.hpp>

#include <vector>


static const double shifts[] = {0.0, -0.5};

#define SEQ_DATATYPE (OpenMPCD::MPCParticlePositionType)



template<typename T>
static void test_computeNormalCoordinates(const T shift)
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


	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
				0, chainCount, d_vectors, d_computationResults),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
				chainLength, chainCount, 0, d_computationResults),
			OpenMPCD::NULLPointerException);

		REQUIRE_THROWS_AS(
			OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
				chainLength, chainCount, d_vectors, 0),
			OpenMPCD::NULLPointerException);
	#endif


	OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
		chainLength,
		chainCount,
		d_vectors,
		d_computationResults,
		shift);

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


	if(shift == 0)
	{ //manually calculated test:
		static const unsigned int chainCount = 4;
		static const unsigned int chainLength = 3;

		static const unsigned int particleCount = chainCount * chainLength;
		static const unsigned int modeCount = chainLength + 1;
		static const unsigned int resultsSize = 3 * chainCount * modeCount;

		static const T positions[] =
			{
				//chain 0:
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				//chain 1:
				0.1, 0.2, 0.4,
				0.3, 0.6, 1.2,
				0.5, 1.0, 1.5,

				//chain 2:
				-10, -11, -12,
				 13,  14,  15,
				-16, -17, -18,

				//chain 3:
				0,  1, -1,
				2,  0, -2,
				0, -3,  3
			};

		static const T expected[] =
			{
				//chain 0:
				  12 / 3,   15 / 3,    18 / 3, //mode 0
				-8.5 / 3, -9.5 / 3, -10.5 / 3, //mode 1
				 4.5 / 3,  4.5 / 3,   4.5 / 3, //mode 2
				-4.0 / 3, -5.0 / 3,  -6.0 / 3, //mode 3

				//chain 1:
				 0.9 / 3,  1.8 / 3,  3.1 / 3, //mode 0
				-0.6 / 3, -1.2 / 3, -1.9 / 3, //mode 1
				 0.3 / 3,  0.6 / 3,  0.7 / 3, //mode 2
				-0.3 / 3, -0.6 / 3, -0.7 / 3,

				//chain 2:
				-13.0 / 3, -14.0 / 3, -15.0 / 3, //mode 0
				  4.5 / 3,   4.5 / 3,   4.5 / 3, //mode 1
				-17.5 / 3, -18.5 / 3, -19.5 / 3, //mode 2
				   39 / 3,    42 / 3,    45 / 3, //mode 3

				//chain 3:
				 2.0 / 3, -2.0 / 3,        0, //mode 0
				-1.0 / 3,  3.5 / 3, -2.5 / 3, //mode 1
				-1.0 / 3, -3.5 / 3,  4.5 / 3, //mode 2
				 2.0 / 3,  2.0 / 3, -4.0 / 3  //mode 3
			};

		T* const d_vectors = dmm.allocateMemory<T>(3 * particleCount);

		//reserve a block before and after the results, to verify that they are
		//not changed
		T* const d_computationResults =	dmm.allocateMemory<T>(3 * resultsSize);

		dmm.copyElementsFromHostToDevice(
			positions, d_vectors, 3 * particleCount);
		dmm.zeroMemory(d_computationResults, 3 * resultsSize);

		OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
			chainLength,
			chainCount,
			d_vectors,
			d_computationResults + resultsSize);

		boost::scoped_array<T> results(new T[3 * resultsSize]);
		dmm.copyElementsFromDeviceToHost(
			d_computationResults, results.get(), 3 * resultsSize);


		for(unsigned int i = 0; i < resultsSize; ++i)
			REQUIRE(results[i] == 0);

		for(unsigned int i = 0; i < resultsSize; ++i)
		{
			CAPTURE(i);
			REQUIRE(results[resultsSize + i] == Approx(expected[i]));
		}

		for(unsigned int i = 0; i < resultsSize; ++i)
			REQUIRE(results[2 * resultsSize + i] == 0);
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::NormalMode::computeNormalCoordinates`",
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
static void test_getAverageNormalCoordinateAutocorrelation()
{
	static const unsigned int chainLength = 11;
	static const unsigned int chainCount = 17;

	static const unsigned int particleCount = chainCount * chainLength;
	static const unsigned int modeCount = chainLength + 1;
	static const unsigned int normalModesSize = 3 * chainCount * modeCount;

	std::vector<OpenMPCD::Vector3D<T> > vectors0;
	std::vector<OpenMPCD::Vector3D<T> > vectorsT;
	std::vector<T> vectors0_raw;
	std::vector<T> vectorsT_raw;
	for(unsigned int i = 0; i < particleCount; ++i)
	{
		OpenMPCD::Vector3D<T> v0(0.1 * i, -0.33 * i, 0.5 + i);
		vectors0.push_back(v0);
		vectors0_raw.push_back(v0.getX());
		vectors0_raw.push_back(v0.getY());
		vectors0_raw.push_back(v0.getZ());

		OpenMPCD::Vector3D<T> vT(0.1 + i, -0.33 + i, 0.5 * i);
		vectorsT.push_back(vT);
		vectorsT_raw.push_back(vT.getX());
		vectorsT_raw.push_back(vT.getY());
		vectorsT_raw.push_back(vT.getZ());
	}


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* const d_vectors0 = dmm.allocateMemory<T>(3 * particleCount);
	T* const d_vectorsT = dmm.allocateMemory<T>(3 * particleCount);
	T* const d_normalModes0 = dmm.allocateMemory<T>(normalModesSize);
	T* const d_normalModesT = dmm.allocateMemory<T>(normalModesSize);

	dmm.copyElementsFromHostToDevice(
		&vectors0_raw[0], d_vectors0, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		&vectorsT_raw[0], d_vectorsT, 3 * particleCount);

	dmm.zeroMemory(d_normalModes0, normalModesSize);
	dmm.zeroMemory(d_normalModesT, normalModesSize);


	OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
		chainLength,
		chainCount,
		d_vectors0,
		d_normalModes0);
	OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
		chainLength,
		chainCount,
		d_vectorsT,
		d_normalModesT);


	using OpenMPCD::CUDA::NormalMode::getAverageNormalCoordinateAutocorrelation;

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			getAverageNormalCoordinateAutocorrelation(
				0, chainCount, d_normalModes0, d_normalModesT),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			getAverageNormalCoordinateAutocorrelation(
				chainLength, chainCount, 0, d_normalModesT),
			OpenMPCD::NULLPointerException);

		REQUIRE_THROWS_AS(
			getAverageNormalCoordinateAutocorrelation(
				chainLength, chainCount, d_normalModes0, 0),
			OpenMPCD::NULLPointerException);
	#endif


	const std::vector<T> result =
		getAverageNormalCoordinateAutocorrelation(
			chainLength,
			chainCount,
			d_normalModes0,
			d_normalModesT);

	REQUIRE(result.size() == modeCount);

	for(unsigned int mode = 0; mode < modeCount; ++mode)
	{
		T sumOfProducts = 0;

		for(unsigned int chain = 0; chain < chainCount; ++chain)
		{
			const OpenMPCD::Vector3D<T>* const positions0 =
				&vectors0[chainLength * chain];
			const OpenMPCD::Vector3D<T>* const positionsT =
				&vectorsT[chainLength * chain];

			const OpenMPCD::Vector3D<T> q0 =
				OpenMPCD::NormalMode::computeNormalCoordinate(
					mode, positions0, chainLength);

			const OpenMPCD::Vector3D<T> qT =
				OpenMPCD::NormalMode::computeNormalCoordinate(
					mode, positionsT, chainLength);

			sumOfProducts += q0.dot(qT);
		}

		const T expected = sumOfProducts / chainCount;

		REQUIRE(expected == Approx(result[mode]));
	}


	{ //manually calculated test; see also test_computeNormalCoordinates above
		static const unsigned int chainCount = 2;
		static const unsigned int chainLength = 3;

		static const unsigned int particleCount = chainCount * chainLength;
		static const unsigned int modeCount = chainLength + 1;
		static const unsigned int resultsSize = 3 * chainCount * modeCount;

		static const T positions0[] =
			{
				//chain 0:
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				//chain 1:
				0.1, 0.2, 0.4,
				0.3, 0.6, 1.2,
				0.5, 1.0, 1.5
			};
		static const T positionsT[] =
			{
				//chain 2:
				-10, -11, -12,
				 13,  14,  15,
				-16, -17, -18,

				//chain 3:
				0,  1, -1,
				2,  0, -2,
				0, -3,  3
			};

		static const T expected[] =
			{
					(-636    - 1.8)  / (3 * 3 * 2),
					(-128.25 + 1.15) / (3 * 3 * 2),
					(-249.75 + 0.75) / (3 * 3 * 2),
					(-636.0  + 1)    / (3 * 3 * 2)
			};


		T* const d_vectors0 = dmm.allocateMemory<T>(3 * particleCount);
		T* const d_vectorsT = dmm.allocateMemory<T>(3 * particleCount);
		T* const d_normalModes0 = dmm.allocateMemory<T>(normalModesSize);
		T* const d_normalModesT = dmm.allocateMemory<T>(normalModesSize);

		dmm.copyElementsFromHostToDevice(
			positions0, d_vectors0, 3 * particleCount);
		dmm.copyElementsFromHostToDevice(
			positionsT, d_vectorsT, 3 * particleCount);

		dmm.zeroMemory(d_normalModes0, normalModesSize);
		dmm.zeroMemory(d_normalModesT, normalModesSize);


		OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
			chainLength,
			chainCount,
			d_vectors0,
			d_normalModes0);
		OpenMPCD::CUDA::NormalMode::computeNormalCoordinates(
			chainLength,
			chainCount,
			d_vectorsT,
			d_normalModesT);


		const std::vector<T> result =
			getAverageNormalCoordinateAutocorrelation(
				chainLength,
				chainCount,
				d_normalModes0,
				d_normalModesT);

		REQUIRE(result.size() == modeCount);
		REQUIRE(result.size() == 4);

		for(unsigned int i = 0; i < modeCount; ++i)
			REQUIRE(result[i] == Approx(expected[i]));
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::NormalMode::getAverageNormalCoordinateAutocorrelation`",
	"[CUDA]")
{
	#define CALLTEST(_r, _data, T) \
		test_getAverageNormalCoordinateAutocorrelation<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}
