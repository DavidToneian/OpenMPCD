/**
 * @file
 * Tests `OpenMPCD::CUDA::BunchIteratorRange`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/BunchIteratorRange.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>


SCENARIO(
	"`OpenMPCD::CUDA::BunchIteratorRange::AdvancingFunctor` on Host",
	"")
{
	typedef float T;
	static const std::size_t arrayLength = 20;

	using OpenMPCD::CUDA::BunchIteratorRange;

	for(unsigned int bunchSize = 0; bunchSize <= 2 * arrayLength; ++bunchSize)
	{
		for(unsigned int gapSize = 0; gapSize <= 2 * arrayLength; ++gapSize)
		{
			typedef BunchIteratorRange<T*>::AdvancingFunctor F;

			if(bunchSize == 0)
			{
				#ifdef OPENMPCD_DEBUG
					REQUIRE_THROWS_AS(
						F(bunchSize, gapSize),
						OpenMPCD::InvalidArgumentException);
				#endif

				continue;
			}

			const F f(bunchSize, gapSize);

			for(unsigned int distance = 0; distance < arrayLength; ++distance)
			{
				unsigned int expected = 0;
				unsigned int positionInBunch = 0;
				const unsigned int d = distance;
				for(unsigned int remaining = d; remaining != 0; --remaining)
				{
					++positionInBunch;
					++expected;

					if(positionInBunch == bunchSize)
					{
						positionInBunch = 0;
						expected += gapSize;
					}
				}

				REQUIRE(expected == f(distance));
			}
		}
	}
}



__global__ static void test_AdvancingFunctor(bool* const result)
{
	typedef float T;
	static const std::size_t arrayLength = 20;

	using OpenMPCD::CUDA::BunchIteratorRange;

	*result = false;

	for(unsigned int bunchSize = 1; bunchSize <= 2 * arrayLength; ++bunchSize)
	{
		for(unsigned int gapSize = 0; gapSize <= 2 * arrayLength; ++gapSize)
		{
			typedef BunchIteratorRange<T*>::AdvancingFunctor F;

			const F f(bunchSize, gapSize);

			for(unsigned int distance = 0; distance < arrayLength; ++distance)
			{
				unsigned int expected = 0;
				unsigned int positionInBunch = 0;
				const unsigned int d = distance;
				for(unsigned int remaining = d; remaining != 0; --remaining)
				{
					++positionInBunch;
					++expected;

					if(positionInBunch == bunchSize)
					{
						positionInBunch = 0;
						expected += gapSize;
					}
				}

				if(expected != f(distance))
				{
					*result = false;
					return;
				}
			}
		}
	}

	*result = true;
}
SCENARIO(
	"`OpenMPCD::CUDA::BunchIteratorRange::AdvancingFunctor` on Device",
	"")
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* const d_result = dmm.allocateMemory<bool>(1);
	dmm.zeroMemory(d_result, 1);

	test_AdvancingFunctor<<<1, 1>>>(d_result);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	bool result = false;
	dmm.copyElementsFromDeviceToHost(d_result, &result, 1);
	REQUIRE(result);
}



SCENARIO(
	"`OpenMPCD::CUDA::BunchIteratorRange`",
	"")
{
	typedef float T;
	static const std::size_t arrayLength = 20;

	using OpenMPCD::CUDA::BunchIteratorRange;

	T array[arrayLength];
	for(std::size_t i = 0; i < arrayLength; ++i)
		array[i] = i + 0.5;

	typedef BunchIteratorRange<T*> Range;


	{ //example from class documentation
		REQUIRE(arrayLength >= 12);

		const Range range(array, array + 12, 3, 2);
		Range::Iterator it = range.begin();

		static const unsigned int expectedIndices[] =
			{0, 1, 2, 5, 6, 7, 10, 11};
		static const unsigned int expectedCount =
			sizeof(expectedIndices) / sizeof(expectedIndices[0]);

		for(std::size_t i = 0; i < expectedCount; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[expectedIndices[i]]);
			++it;
		}

		REQUIRE(it == range.end());
	}


	for(unsigned int bunchSize = 0; bunchSize <= 2 * arrayLength; ++bunchSize)
	{
		for(unsigned int gapSize = 0; gapSize <= 2 * arrayLength; ++gapSize)
		{
			if(bunchSize == 0)
			{
				#ifdef OPENMPCD_DEBUG
					REQUIRE_THROWS_AS(
						Range(array, array + arrayLength, bunchSize, gapSize),
						OpenMPCD::InvalidArgumentException);
				#endif

				continue;
			}

			const Range range(array, array + arrayLength, bunchSize, gapSize);

			Range::Iterator it = range.begin();
			unsigned int i = 0;
			unsigned int positionInBunch = 0;
			for(;;)
			{
				REQUIRE(it != range.end());
				REQUIRE(*it == array[i]);

				++it;
				++i;
				++positionInBunch;
				if(positionInBunch == bunchSize)
				{
					positionInBunch = 0;
					i += gapSize;
				}

				if(i >= arrayLength)
					break;
			}

			REQUIRE(it == range.end());
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::BunchIteratorRange` usage with `thrust` on Host",
	"")
{
	typedef float T;
	static const std::size_t arrayLength = 20;

	using OpenMPCD::CUDA::BunchIteratorRange;

	T array[arrayLength];
	for(std::size_t i = 0; i < arrayLength; ++i)
		array[i] = i + 0.5;

	typedef BunchIteratorRange<T*> Range;


	{ //example from class documentation
		REQUIRE(arrayLength >= 12);

		static const unsigned int expectedIndices[] =
			{0, 1, 2, 5, 6, 7, 10, 11};
		static const unsigned int expectedCount =
			sizeof(expectedIndices) / sizeof(expectedIndices[0]);

		T expectedSum = 0;
		for(std::size_t i = 0; i < expectedCount; ++i)
			expectedSum += array[expectedIndices[i]];


		const Range range(array, array + 12, 3, 2);
		const T sum = thrust::reduce(thrust::host, range.begin(), range.end());
		REQUIRE(sum == expectedSum);
	}


	for(unsigned int bunchSize = 1; bunchSize <= 2 * arrayLength; ++bunchSize)
	{
		for(unsigned int gapSize = 0; gapSize <= 2 * arrayLength; ++gapSize)
		{
			const Range range(array, array + arrayLength, bunchSize, gapSize);

			T expectedSum = 0;

			for(Range::Iterator it = range.begin(); it != range.end(); ++it)
				expectedSum += *it;

			const T sum =
				thrust::reduce(thrust::host, range.begin(), range.end());

			REQUIRE(sum == expectedSum);
		}
	}
}



SCENARIO(
	"`OpenMPCD::CUDA::BunchIteratorRange` usage with `thrust` on Device",
	"[CUDA]")
{
	typedef float T;
	static const std::size_t arrayLength = 20;

	using OpenMPCD::CUDA::BunchIteratorRange;

	T array[arrayLength];
	for(std::size_t i = 0; i < arrayLength; ++i)
		array[i] = i + 0.5;


	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* const d_array = dmm.allocateMemory<T>(arrayLength);
	dmm.copyElementsFromHostToDevice(array, d_array, arrayLength);

	typedef BunchIteratorRange<T*> Range;

	{ //example from class documentation
		REQUIRE(arrayLength >= 12);

		static const unsigned int expectedIndices[] =
			{0, 1, 2, 5, 6, 7, 10, 11};
		static const unsigned int expectedCount =
			sizeof(expectedIndices) / sizeof(expectedIndices[0]);

		T expectedSum = 0;
		for(std::size_t i = 0; i < expectedCount; ++i)
			expectedSum += array[expectedIndices[i]];


		const Range d_range(d_array, d_array + 12, 3, 2);
		const T sum =
			thrust::reduce(thrust::device, d_range.begin(), d_range.end());
		REQUIRE(sum == expectedSum);
	}


	for(unsigned int bunchSize = 1; bunchSize <= 2 * arrayLength; ++bunchSize)
	{
		for(unsigned int gapSize = 0; gapSize <= 2 * arrayLength; ++gapSize)
		{
			const Range range(
				array, array + arrayLength, bunchSize, gapSize);
			const Range d_range(
				d_array, d_array + arrayLength, bunchSize, gapSize);

			T expectedSum = 0;

			for(Range::Iterator it = range.begin(); it != range.end(); ++it)
				expectedSum += *it;

			const T sum =
				thrust::reduce(thrust::device, d_range.begin(), d_range.end());

			REQUIRE(sum == expectedSum);
		}
	}
}
