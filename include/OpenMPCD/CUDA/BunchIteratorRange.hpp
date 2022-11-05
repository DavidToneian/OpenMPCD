/**
 * @file
 * Defines the `OpenMPCD::CUDA::BunchIteratorRange` class.
 */

#ifndef OPENMPCD_CUDA_BUNCHITERATORRANGE_HPP
#define OPENMPCD_CUDA_BUNCHITERATORRANGE_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Wraps an iterator range such that it is grouped into "bunches" of a given
 * length, with a configurable gap between bunches.
 *
 * As an example, assume that one has an input iterator range that represents
 * the sequence `A_0`, `A_1`, `A_2`, ..., `A_N`.
 * If `bunchSize` is set to `3`, `gapSize` is set to `2`, and `N = 11` (i.e. if
 * the input sequence has `12` elements), this iterator range will iterate over
 * the elements `A_0`, `A_1`, `A_2`, followed by `A_5`, `A_6`, `A_7` (skipping
 * the two elements `A_3` and `A_4`), followed by `A_10` and `A_11`, and end
 * there since the input sequence is exhausted.
 *
 * @tparam UnderlyingIterator
 *         The underlying iterator type.
 */
template<typename UnderlyingIterator>
class BunchIteratorRange
{

public:
	typedef
		typename thrust::iterator_difference<UnderlyingIterator>::type
		IteratorDifference;
		///< The type of the difference of two of the underlying iterators.

	/**
	 * Functor class for advancing a pointer by the given number of effective
	 * elements.
	 */
	class AdvancingFunctor
		: public thrust::unary_function<IteratorDifference, IteratorDifference>
	{
		public:
			/**
			 * The constructor.
			 *
			 * @throw OpenMPCD::InvalidArgumentException
			 *        If `OPENMPCD_DEBUG` is defined, throws if
			 *        `bunchSize_ == 0`.
			 *
			 * @param[in] bunchSize_ The bunch size, which must not be `0`.
			 * @param[in] gapSize_   The gap size.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			AdvancingFunctor(
				const unsigned int bunchSize_, const unsigned int gapSize_)
				: bunchSize(bunchSize_), gapSize(gapSize_)
			{
				OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
					bunchSize != 0, InvalidArgumentException);
			}

			/**
			 * Calculates how many times the underlying iterator has to be
			 * incremented to achieve `i` increments of the effective iterator.
			 *
			 * @param[in] i
			 *            The number of times the effective iterator is to be
			 *            incremented.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const IteratorDifference operator()(
				const IteratorDifference& i) const
			{
				OPENMPCD_DEBUG_ASSERT(i >= 0);

				unsigned int ret = 0;
				unsigned int positionInBunch = 0;
				for(unsigned int remaining = i; remaining != 0; --remaining)
				{
					++positionInBunch;
					++ret;

					if(positionInBunch == bunchSize)
					{
						positionInBunch = 0;
						ret += gapSize;
					}
				}

				return ret;
			}

		private:
			const unsigned int bunchSize; ///< The bunch size.
			const unsigned int gapSize; ///< The gap size.
	};

	typedef
		typename thrust::counting_iterator<IteratorDifference>
		CountingIterator;
		///< Type that is used to linearly increment a count.
	typedef
		typename thrust::transform_iterator<AdvancingFunctor, CountingIterator>
		TransformIterator;
		///< Type that applies `AdvancingFunctor` to the input sequence.
	typedef
		typename thrust::permutation_iterator<
			UnderlyingIterator, TransformIterator>
		PermutationIterator;
		/**< Type that will be indexing into the `UnderlyingIterator` according
		     to the `TransformIterator`.*/

	typedef PermutationIterator Iterator; ///< The effective iterator type.

public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `bunchSize_ == 0`.
	 *
	 * @param[in] start       The first iterator to iterate over.
	 * @param[in] pastTheEnd_ The first iterator that is past-the-end.
	 * @param[in] bunchSize_  The bunch size, which must not be 0.
	 * @param[in] gapSize_    The gap size.
	 */
	BunchIteratorRange(
		const UnderlyingIterator& start, const UnderlyingIterator& pastTheEnd_,
		const unsigned int bunchSize_, const unsigned int gapSize_)
		: first(start), pastTheEnd(pastTheEnd_),
		  bunchSize(bunchSize_), gapSize(gapSize_)
	{
		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
			bunchSize != 0, InvalidArgumentException);
	}

public:
	/**
	 * Returns the first iterator in the effective range.
	 */
	Iterator begin() const
	{
		return Iterator(
			first,
			TransformIterator(
				CountingIterator(0),
				AdvancingFunctor(bunchSize, gapSize)));
	}

	/**
	 * Returns the first past-the-end iterator of the effective range.
	 */
	Iterator end() const
	{
		IteratorDifference effectiveCount = 0;
		IteratorDifference underlyingRemaining = pastTheEnd - first;

		for(;;)
		{
			if(underlyingRemaining <= bunchSize)
			{
				effectiveCount += underlyingRemaining;
				break;
			}

			effectiveCount += bunchSize;
			underlyingRemaining -= bunchSize;

			if(underlyingRemaining <= gapSize)
				break;

			underlyingRemaining -= gapSize;
		}

		return begin() + effectiveCount;
	}

private:
	UnderlyingIterator first;      ///< The first iterator to iterate over.
	UnderlyingIterator pastTheEnd; ///< The first iterator that is past-the-end.
	const unsigned int bunchSize;  ///< The bunch size.
	const unsigned int gapSize;    ///< The gap size.

}; //class BunchIteratorRange


} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_BUNCHITERATORRANGE_HPP
