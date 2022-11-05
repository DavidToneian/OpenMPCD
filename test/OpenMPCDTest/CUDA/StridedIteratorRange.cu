/**
 * @file
 * Tests `OpenMPCD::CUDA::StridedIteratorRange`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/StridedIteratorRange.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>

#include <boost/type_traits/is_same.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>


SCENARIO(
	"`OpenMPCD::CUDA::StridedIteratorRange`, static stride",
	"")
{
	typedef float T;
	static const std::size_t arrayLength = 10;

	using OpenMPCD::CUDA::StridedIteratorRange;

	T array[arrayLength];
	for(std::size_t i = 0; i < arrayLength; ++i)
		array[i] = i + 0.5;

	REQUIRE(arrayLength == 10);

	{
		typedef StridedIteratorRange<T*, 1> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 10; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 2> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 5; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[2 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 3> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 4; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 5> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 2; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[5 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 6> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 2; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[6 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 9> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 2; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[9 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 10> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 1; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 11> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 1; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		typedef StridedIteratorRange<T*, 12> Range;

		const Range range(array, array + arrayLength);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 1; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::StridedIteratorRange`, dynamic stride",
	"")
{
	typedef float T;
	static const std::size_t arrayLength = 10;

	using OpenMPCD::CUDA::StridedIteratorRange;

	T array[arrayLength];
	for(std::size_t i = 0; i < arrayLength; ++i)
		array[i] = i + 0.5;

	REQUIRE(arrayLength == 10);

	typedef StridedIteratorRange<T*> Range;
	BOOST_STATIC_ASSERT(
		(boost::is_same<Range, StridedIteratorRange<T*, 0> >::value));

	{
		const Range range(array, array + arrayLength, 1);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 10; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 2);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 5; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[2 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 3);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 4; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 5);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 2; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[5 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 6);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 2; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[6 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 9);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 2; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[9 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 10);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 1; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 11);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 1; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}

	{
		const Range range(array, array + arrayLength, 12);
		Range::Iterator it = range.begin();

		for(std::size_t i = 0; i < 1; ++i)
		{
			REQUIRE(it != range.end());
			REQUIRE(*it == array[3 * i]);
			++it;
		}

		REQUIRE(it == range.end());
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::StridedIteratorRange`, static stride, "
		"application via `thrust::reduce`",
	"[CUDA]")
{
	typedef double T;
	static const std::size_t arrayLength = 10;

	using OpenMPCD::CUDA::StridedIteratorRange;

	T array[arrayLength];
	for(std::size_t i = 0; i < arrayLength; ++i)
		array[i] = i + 0.5;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	T* d_array;
	dmm.allocateMemory(&d_array, arrayLength);
	dmm.copyElementsFromHostToDevice(array, d_array, arrayLength);


	T expectedResults[11] = {0};
	for(std::size_t stride = 1; stride <= 11; ++stride)
	{
		for(std::size_t offset = 0; offset < arrayLength; offset += stride)
			expectedResults[stride - 1] += array[offset];
	}


	const StridedIteratorRange<T*, 1> range1(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 2> range2(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 3> range3(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 4> range4(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 5> range5(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 6> range6(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 7> range7(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 8> range8(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 9> range9(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 10> range10(d_array, d_array + arrayLength);
	const StridedIteratorRange<T*, 11> range11(d_array, d_array + arrayLength);

	const T result1 =
		thrust::reduce(thrust::device, range1.begin(), range1.end());
	const T result2 =
		thrust::reduce(thrust::device, range2.begin(), range2.end());
	const T result3 =
		thrust::reduce(thrust::device, range3.begin(), range3.end());
	const T result4 =
		thrust::reduce(thrust::device, range4.begin(), range4.end());
	const T result5 =
		thrust::reduce(thrust::device, range5.begin(), range5.end());
	const T result6 =
		thrust::reduce(thrust::device, range6.begin(), range6.end());
	const T result7 =
		thrust::reduce(thrust::device, range7.begin(), range7.end());
	const T result8 =
		thrust::reduce(thrust::device, range8.begin(), range8.end());
	const T result9 =
		thrust::reduce(thrust::device, range9.begin(), range9.end());
	const T result10 =
		thrust::reduce(thrust::device, range10.begin(), range10.end());
	const T result11 =
		thrust::reduce(thrust::device, range11.begin(), range11.end());


	REQUIRE(expectedResults[1 - 1] == result1);
	REQUIRE(expectedResults[2 - 1] == result2);
	REQUIRE(expectedResults[3 - 1] == result3);
	REQUIRE(expectedResults[4 - 1] == result4);
	REQUIRE(expectedResults[5 - 1] == result5);
	REQUIRE(expectedResults[6 - 1] == result6);
	REQUIRE(expectedResults[7 - 1] == result7);
	REQUIRE(expectedResults[8 - 1] == result8);
	REQUIRE(expectedResults[9 - 1] == result9);
	REQUIRE(expectedResults[10 - 1] == result10);
	REQUIRE(expectedResults[11 - 1] == result11);
}
