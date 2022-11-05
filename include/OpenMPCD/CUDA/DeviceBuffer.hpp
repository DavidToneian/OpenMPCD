/**
 * @file
 * Defines the `OpenMPCD::CUDA::DeviceBuffer` class.
 */

#ifndef OPENMPCD_CUDA_DEVICEBUFFER_HPP
#define OPENMPCD_CUDA_DEVICEBUFFER_HPP

#include <cstddef>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Represents a memory buffer on the Device.
 *
 * The memory for this buffer will be allocated and freed automatically.
 *
 * @tparam T The underlying data type.
 */
template<typename T>
class DeviceBuffer
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `elemtCount_ == 0`.
	 *
	 * @param[in] elementCount_
	 *            The number of elements to allocate memory for. No constructors
	 *            will be called. Must not be `0`.
	 */
	DeviceBuffer(const std::size_t elementCount_);

private:
	DeviceBuffer(const DeviceBuffer&); ///< The copy constructor.

public:
	/**
	 * The destructor.
	 *
	 * This will free the allocated memory, but will not call any destructors.
	 */
	~DeviceBuffer();

public:
	/**
	 * Returns the Device pointer.
	 */
	T* getPointer();

	/**
	 * Returns the Device pointer.
	 */
	const T* getPointer() const;

	/**
	 * Returns the number of elements the buffer can store.
	 */
	std::size_t getElementCount() const;

	/**
	 * Copies `getElementCount()` elements from the given Device pointer to this
	 * buffer.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `src == nullptr`.
	 *
	 * @param[in] src
	 *           The Device pointer to copy from, which must not be `nullptr`
	 *           and must point to at least `getElementCount()` elements.
	 */
	void copyFromDevice(const T* const src);

	/**
	 * Writes zero-bytes to the entirety of this buffer.
	 */
	void zeroMemory();

private:
	const DeviceBuffer& operator=(const DeviceBuffer&);
		///< The assignment operator.

private:
	T* pointer;               ///< The Device pointer.
	std::size_t elementCount; ///< The number of elements that can be stored.

}; //class DeviceBuffer
} //namespace CUDA
} //namespace OpenMPCD


#include <OpenMPCD/CUDA/ImplementationDetails/DeviceBuffer.hpp>

#endif //OPENMPCD_CUDA_DEVICEBUFFER_HPP
