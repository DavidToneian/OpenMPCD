/**
 * @file
 * Defines the OpenMPCD::StridedPointerIterator class.
 */

#ifndef OPENMPCD_STRIDEDPOINTERITERATOR_HPP
#define OPENMPCD_STRIDEDPOINTERITERATOR_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>

#include <boost/static_assert.hpp>
#include <iterator>

namespace OpenMPCD
{
	/**
	 * Wraps a pointer in such a way that incrementing this iterator is equivalent to
	 * incrementing the underlying pointer once or more times, depending on the stride specified.
	 * @tparam Pointee The type the underlying pointer points at.
	 * @tparam stride  The iteration stride, which must not be 0.
	 */
	template<typename Pointee, unsigned int stride> class StridedPointerIterator
	{
		BOOST_STATIC_ASSERT(stride != 0);

		public:
			/**
			 * The default constructor.
			 * The constructed instance is singular.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			StridedPointerIterator() : current(NULL), pastEnd(NULL)
			{
			}

			/**
			 * The constructor.
			 * @param[in] start    The first element to iterate over.
			 * @param[in] pastEnd_ The first element that is past the end of the array to iterate over.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			StridedPointerIterator(Pointee* const start, const Pointee* const pastEnd_) : current(start), pastEnd(pastEnd_)
			{
			}

			/**
			 * The copy constructor.
			 * @param[in] rhs The right-hand-side instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			StridedPointerIterator(const StridedPointerIterator& rhs)
				: current(rhs.current), pastEnd(rhs.pastEnd)
			{
			}

		public:
			/**
			 * Returns whether this instance is singular, i.e. invalid.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool isSingular() const
			{
				return current==NULL;
			}

			/**
			 * Returns whether this iterator is past-the-end.
			 * @throw InvalidCallException Throws if <c>isSingular()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool isPastTheEnd() const
			{
				#ifndef __CUDA_ARCH__
					if(isSingular())
						OPENMPCD_THROW(InvalidCallException, "isPastTheEnd()");
				#endif

				return current == pastEnd;
			}

			/**
			 * Returns whether this instance can be dereferenced.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool isDereferenceable() const
			{
				if(isSingular())
					return false;

				if(isPastTheEnd())
					return false;

				return true;
			}

			/**
			 * Returns whether this instance is valid.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool isValid() const
			{
				return isDereferenceable() || isPastTheEnd();
			}

			/**
			 * Returns whether this instance is incrementable.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool isIncrementable() const
			{
				if(isSingular())
					return false;

				if(isPastTheEnd())
					return false;

				return true;
			}

		public:
			/**
			 * Dereferencing operator.
			 * @throw InvalidCallException Throws if <c>!isDereferenceable()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			Pointee& operator*()
			{
				#ifndef __CUDA_ARCH__
					if(!isDereferenceable())
						OPENMPCD_THROW(InvalidCallException, "operator*");
				#endif

				return *current;
			}

			/**
			 * Dereferencing operator.
			 * @throw InvalidCallException Throws if <c>!isDereferenceable()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Pointee& operator*() const
			{
				#ifndef __CUDA_ARCH__
					if(!isDereferenceable())
						OPENMPCD_THROW(InvalidCallException, "operator*");
				#endif

				return current;
			}

			/**
			 * Member access operator.
			 * @throw InvalidCallException Throws if <c>!isDereferenceable()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			Pointee* operator->()
			{
				#ifndef __CUDA_ARCH__
					if(!isDereferenceable())
						OPENMPCD_THROW(InvalidCallException, "operator->");
				#endif

				return current;
			}

			/**
			 * Member access operator.
			 * @throw InvalidCallException Throws if <c>!isDereferenceable()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Pointee* operator->() const
			{
				#ifndef __CUDA_ARCH__
					if(!isDereferenceable())
						OPENMPCD_THROW(InvalidCallException, "operator->");
				#endif

				return current;
			}

			/**
			 * The prefix increment operator.
			 * @throw InvalidCallException Throws if <c>!isDereferenceable()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			StridedPointerIterator& operator++()
			{
				#ifndef __CUDA_ARCH__
					if(!isDereferenceable())
						OPENMPCD_THROW(InvalidCallException, "operator++");
				#endif

				for(unsigned int i=0; i<stride; ++i)
				{
					++current;

					if(current==pastEnd)
						break;
				}

				return *this;
			}

			/**
			 * The postfix increment operator.
			 * @throw InvalidCallException Throws if <c>!isDereferenceable()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			StridedPointerIterator operator++(int)
			{
				#ifndef __CUDA_ARCH__
					if(!isDereferenceable())
						OPENMPCD_THROW(InvalidCallException, "operator++");
				#endif

				StridedPointerIterator ret=*this;

				for(unsigned int i=0; i<stride; ++i)
				{
					++current;

					if(current==pastEnd)
						break;
				}

				return ret;
			}

			/**
			 * Advances the pointer n times, or until the past-the-end iterator is reached, whichever comes first.
			 * @param[in] n The number of times to advance the pointer.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const StridedPointerIterator& operator+=(const unsigned int n)
			{
				for(unsigned int i=0; i<n; ++i)
				{
					operator++();

					if(isPastTheEnd())
						break;
				}

				return *this;
			}

			/**
			 * Returns whether the address this instance and rhs point to the same memory.
			 * This function does not consider whether both this instance and rhs have the same iteration range.
			 * @param[in] rhs The right-hand-side instance.
			 * @throw InvalidCallException Throws if <c>!isValid()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool operator==(const StridedPointerIterator& rhs) const
			{
				#ifndef __CUDA_ARCH__
					if(!isValid())
						OPENMPCD_THROW(InvalidCallException, "operator==");
				#endif

				return current == rhs.current;
			}

			/**
			 * Returns whether the address this instance and rhs point to different memory.
			 * This function does not consider whether both this instance and rhs have the same iteration range.
			 * @param[in] rhs The right-hand-side instance.
			 * @throw InvalidCallException Throws if <c>!isValid()</c>.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool operator!=(const StridedPointerIterator& rhs) const
			{
				#ifndef __CUDA_ARCH__
					if(!isValid())
						OPENMPCD_THROW(InvalidCallException, "operator!=");
				#endif

				return current != rhs.current;
			}

		private:
			Pointee* current;       ///< The element this iterator points at.
			const Pointee* pastEnd; ///< A pointer that points one element past the end of the array to iterate over.
	};
}

namespace std
{
	/**
	 * Specialisation of the std::iterator_traits template for StridedPointerIterator.
	 * @tparam Pointee The type the underlying pointer points at.
	 * @tparam stride  The iteration stride, which must not be 0.
	 */
	template<typename Pointee, unsigned int stride>
		struct iterator_traits<OpenMPCD::StridedPointerIterator<Pointee, stride> >
	{
		typedef std::ptrdiff_t difference_type; ///< Type for distance between iterators.

		typedef Pointee        value_type; ///< Type iterated over.
		typedef value_type*    pointer;    ///< Pointer to the type iterated over.
		typedef value_type&    reference;  ///< Reference to the type iterated over.

		typedef std::input_iterator_tag iterator_category; ///< Specifies the iterator category.
	};
}

#endif
