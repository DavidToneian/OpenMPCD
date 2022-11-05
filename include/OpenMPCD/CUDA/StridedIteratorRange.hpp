/**
 * @file
 * Defines the `OpenMPCD::CUDA::StridedIteratorRange` class.
 */

#ifndef OPENMPCD_CUDA_STRIDEDITERATORRANGE_HPP
#define OPENMPCD_CUDA_STRIDEDITERATORRANGE_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/static_assert.hpp>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Wraps an iterator range such that the iteration stride is not (necessarily)
 * `1`.
 *
 * This strided iterator range exposes the underlying iterator range in such a
 * way that incrementing the exposed iterator (called "effective iterator") once
 * is equivalent to incrementing the underlying iterator as many times as the
 * `stride` specifies.
 *
 * @remark
 * This code has been adapted from `examples/strided_range.cu` of the `thrust`
 * library.
 *
 * @tparam UnderlyingIterator
 *         The underlying iterator type.
 * @tparam stride
 *         The iterator stride. This parameter specifies how often the
 *         underlying iterator gets incremented each time this instance of
 *         `StridedIteratorRange` gets incremented.
 *         The special value `0` selects a (partially) specialized template of
 *         `StridedIteratorRange` that allows one to set the stride in the
 *         constructor.
 */
template<typename UnderlyingIterator, unsigned int stride = 0>
class StridedIteratorRange
{

BOOST_STATIC_ASSERT(stride != 0);

public:
	typedef
		typename thrust::iterator_difference<UnderlyingIterator>::type
		IteratorDifference;
		///< The type of the difference of two of the underlying iterators.

	/**
	 * Functor class for advancing a pointer by the given stride.
	 */
	class StrideFunctor
		: public thrust::unary_function<IteratorDifference, IteratorDifference>
	{
		public:
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
				return stride * i;
			}
	};

	typedef
		typename thrust::counting_iterator<IteratorDifference>
		CountingIterator;
		///< Type that is used to linearly increment a count.
	typedef
		typename thrust::transform_iterator<StrideFunctor, CountingIterator>
		TransformIterator;
		///< Type that applies `StrideFunctor` to the input sequence.
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
	 * @param[in] start       The first iterator to iterate over.
	 * @param[in] pastTheEnd_ The first iterator that is past-the-end.
	 */
	StridedIteratorRange(
		const UnderlyingIterator& start, const UnderlyingIterator& pastTheEnd_)
		: first(start), pastTheEnd(pastTheEnd_)
	{
	}

public:
	/**
	 * Returns the first iterator in the strided range.
	 */
	Iterator begin() const
	{
		return Iterator(
			first, TransformIterator(CountingIterator(0), StrideFunctor()));
	}

	/**
	 * Returns the first past-the-end iterator of the strided range.
	 */
	Iterator end() const
	{
		return begin() + ((pastTheEnd - first) + (stride - 1)) / stride;
	}

private:
	UnderlyingIterator first;      ///< The first iterator to iterate over.
	UnderlyingIterator pastTheEnd; ///< The first iterator that is past-the-end.

}; //class StridedIteratorRange


/**
 * Partial template specialization of `StridedIteratorRange` for dynamic
 * strides.
 */
template<typename UnderlyingIterator>
class StridedIteratorRange<UnderlyingIterator, 0>
{

public:
	typedef
		typename thrust::iterator_difference<UnderlyingIterator>::type
		IteratorDifference;
		///< The type of the difference of two of the underlying iterators.

	/**
	 * Functor class for advancing a pointer by the given stride.
	 */
	class StrideFunctor
		: public thrust::unary_function<IteratorDifference, IteratorDifference>
	{
		public:
			/**
			 * The constructor.
			 *
			 * @param[in] s The iterator stride.
			 */
			StrideFunctor(const unsigned int s) : stride(s)
			{
			}

			/**
			 * Calculates how many times the underlying iterator has to be
			 * incremented to achieve `i` increments of the effective iterator.
			 *
			 * @param[in] i
			 *            The number of times the effective iterator is to be
			 *            incremented.
			 */
			const IteratorDifference operator()(
				const IteratorDifference& i) const
			{
				return stride * i;
			}

		private:
			const unsigned int stride; ///< The iterator stride.
	};

	typedef
		typename thrust::counting_iterator<IteratorDifference>
		CountingIterator;
		///< Type that is used to linearly increment a count.
	typedef
		typename thrust::transform_iterator<StrideFunctor, CountingIterator>
		TransformIterator;
		///< Type that applies `StrideFunctor` to the input sequence.
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
	 * @param[in] start       The first iterator to iterate over.
	 * @param[in] pastTheEnd_ The first iterator that is past-the-end.
	 * @param[in] stride_     The iterator stride.
	 */
	StridedIteratorRange(
		const UnderlyingIterator& start, const UnderlyingIterator& pastTheEnd_,
		const unsigned int stride_)
		: first(start), pastTheEnd(pastTheEnd_), stride(stride_)
	{
	}

public:
	/**
	 * Returns the first iterator in the strided range.
	 */
	Iterator begin() const
	{
		return Iterator(
			first,
			TransformIterator(CountingIterator(0), StrideFunctor(stride)));
	}

	/**
	 * Returns the first past-the-end iterator of the strided range.
	 */
	Iterator end() const
	{
		return begin() + ((pastTheEnd - first) + (stride - 1)) / stride;
	}

private:
	UnderlyingIterator first;      ///< The first iterator to iterate over.
	UnderlyingIterator pastTheEnd; ///< The first iterator that is past-the-end.
	const unsigned int stride;     ///< The iterator stride.

}; //class StridedIteratorRange


} //namespace CUDA
} //namespace OpenMPCD

#endif
