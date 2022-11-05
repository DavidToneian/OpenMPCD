/**
 * @file
 * Defines the OpenMPCD::StridedPointerIteratorRange class.
 */

#ifndef OPENMPCD_STRIDEDPOINTERITERATORRANGE_HPP
#define OPENMPCD_STRIDEDPOINTERITERATORRANGE_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/StridedPointerIterator.hpp>

#include <boost/static_assert.hpp>

namespace OpenMPCD
{
	/**
	 * Represents a range of pointers that can be iterated over.
	 * @tparam Pointee The type the underlying pointer points at.
	 * @tparam stride  The iteration stride, which must not be 0.
	 */
	template<typename Pointee, unsigned int stride> class StridedPointerIteratorRange
	{
		BOOST_STATIC_ASSERT(stride != 0);

		public:
			/**
			 * The constructor.
			 * @param[in] start_           The first element to iterate over.
			 * @param[in] numberOfElements The total number of elements in the array.
			 */
			StridedPointerIteratorRange(Pointee* const start_, const std::size_t numberOfElements)
				: start(start_), pastTheEnd(start + numberOfElements)
			{
			}

		public:
			/**
			 * Returns the first iterator in the range.
			 */
			StridedPointerIterator<Pointee, stride> begin() const
			{
				return StridedPointerIterator<Pointee, stride>(start, pastTheEnd);
			}

			/**
			 * Returns the past-the-end iterator.
			 */
			StridedPointerIterator<Pointee, stride> end() const
			{
				return StridedPointerIterator<Pointee, stride>(pastTheEnd, pastTheEnd);
			}

		private:
			Pointee* start;      ///< The first pointer in the range.
			Pointee* pastTheEnd; ///< The first pointer that lies past the end of the range.
	};
}

#endif
