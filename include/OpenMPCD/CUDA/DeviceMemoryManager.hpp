/**
 * @file
 * Defines the OpenMPCD::CUDA::DeviceMemoryManager class.
 */

#ifndef OPENMPCD_CUDA_DEVICEMEMORYMANAGER_HPP
#define OPENMPCD_CUDA_DEVICEMEMORYMANAGER_HPP

#include <OpenMPCD/Exceptions.hpp>

#include <set>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Class for managing memory on the CUDA Device.
 */
class DeviceMemoryManager
{
	public:
		/**
		 * The constructor.
		 * The autofree feature is disabled by default.
		 */
		DeviceMemoryManager();

	private:
		DeviceMemoryManager(const DeviceMemoryManager&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 * If there are Device buffers which have not been freed,
		 * and the autofree flag has not been set,
		 * prints corresponding warning messages to `stderr`.
		 */
		~DeviceMemoryManager();

	public:
		/**
		 * Allocates Device memory for the given number of instances of the supplied type.
		 * The returned pointer points to at least instanceCount * sizeof(Pointee) bytes of allocated Device memory.
		 *
		 * Memory allocated through this function is registered with this
		 * instance. Unless the memory is freed via a call to this instance's
		 * `freeMemory`, or the autofree flag is set via `setAutofree`, this
		 * instance will, during destruction, print error messages about every
		 * unfreed allocation in the destructor.
		 *
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the buffer could not be allocated.
		 *
		 * @tparam Pointee The pointee type.
		 *
		 * @param[in] instanceCount The number of instances the returned buffer
		 *                          should fit at a minimum. If `0`, `nullptr`
		 *                          is returned.
		 */
		template<typename Pointee> Pointee* allocateMemory(const unsigned int instanceCount)
		{
			return static_cast<Pointee*>(allocateMemoryInternal(instanceCount * sizeof(Pointee)));
		}

		/**
		 * Allocates Device memory for the given number of instances of the supplied type.
		 * After a successful call, the given pointer points to at least instanceCount * sizeof(Pointee) bytes
		 * of allocated Device memory.
		 *
		 * Memory allocated through this function is registered with this
		 * instance. Unless the memory is freed via a call to this instance's
		 * `freeMemory`, or the autofree flag is set via `setAutofree`, this
		 * instance will, during destruction, print error messages about every
		 * unfreed allocation in the destructor.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If `OPENMPCD_DEBUG` is defined, throws if `pointerToPointer` is
		 *        `nullptr`.
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the buffer could not be allocated.
		 *
		 * @tparam Pointee The pointee type.
		 *
		 * @param[out] pointerToPointer A valid pointer to the pointer that is
		 *                              to be set. Will be set to `nullptr` if
		 *                              `instanceCount == 0`.
		 * @param[in]  instanceCount    The number of instances the returned buffer should fit at a minimum.
		 */
		template<typename Pointee> void allocateMemory(Pointee** pointerToPointer, const unsigned int instanceCount)
		{
			#ifdef OPENMPCD_DEBUG
				if(pointerToPointer == NULL)
					OPENMPCD_THROW(NULLPointerException, "pointerToPointer");
			#endif

			*pointerToPointer = allocateMemory<Pointee>(instanceCount);
		}

		/**
		 * Allocates Device memory for the given number of instances of the
		 * supplied type.
		 * The returned pointer points to at least
		 * `instanceCount * sizeof(Pointee)` bytes of allocated Device memory.
		 *
		 * Memory allocated through this function is not registered with this
		 * instance, in contrast to `allocateMemory`.
		 *
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the buffer could not be allocated.
		 *
		 * @tparam Pointee The pointee type.
		 *
		 * @param[in] instanceCount The number of instances the returned buffer
		 *                          should fit at a minimum. If `0`, `nullptr`
		 *                          is returned.
		 */
		template<typename Pointee>
		static
		Pointee* allocateMemoryUnregistered(const unsigned int instanceCount)
		{
			return
				static_cast<Pointee*>(
					allocateMemoryInternalUnregistered(
						instanceCount * sizeof(Pointee)));
		}

		/**
		 * Allocates Device memory for the given number of instances of the
		 * supplied type.
		 * After a successful call, the given pointer points to at least
		 * `instanceCount * sizeof(Pointee)` bytes of allocated Device memory.
		 *
		 * Memory allocated through this function is not registered with this
		 * instance, in contrast to `allocateMemory`.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If `OPENMPCD_DEBUG` is defined, throws if `pointerToPointer` is
		 *        `nullptr`.
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the buffer could not be allocated.
		 *
		 * @tparam Pointee The pointee type.
		 *
		 * @param[out] pointerToPointer A valid pointer to the pointer that is
		 *                              to be set. Will be set to `nullptr` if
		 *                              `instanceCount == 0`.
		 * @param[in]  instanceCount    The number of instances the returned
		 *                              buffer should fit at a minimum.
		 */
		template<typename Pointee>
		static
		void allocateMemoryUnregistered(
			Pointee** pointerToPointer, const unsigned int instanceCount)
		{
			#ifdef OPENMPCD_DEBUG
				if(pointerToPointer == NULL)
					OPENMPCD_THROW(NULLPointerException, "pointerToPointer");
			#endif

			*pointerToPointer =
				allocateMemoryUnregistered<Pointee>(instanceCount);
		}

		/**
		 * Frees the Device memory pointed to by the given pointer.
		 *
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the given non-`nullptr` pointer has not been
		 *        allocated through and registered with this instance, i.e. if
		 *        the given pointer has not been created via a call to
		 *        `allocateMemory`.
		 *
		 * @param[in] pointer The pointer to free. If it is `nullptr`, nothing
		 *                    happens.
		 */
		void freeMemory(void* const pointer);

		/**
		 * Frees the Device memory pointed to by the given pointer.
		 *
		 * Note that one should not use this function with pointers allocated
		 * in a way that the pointer is registered with an instance of this
		 * class, e.g. via `allocateMemory`, since otherwise, that memory is
		 * either going to be freed twice, or a (spurious) warning about unfreed
		 * memory may be produced.
		 *
		 * @throw OpenMPCD::MemoryManagementException
		 *        If `OPENMPCD_DEBUG` is defined, throws if
		 *        `pointer != nullptr && !isDeviceMemoryPointer(pointer)`.
		 *
		 * @param[in] pointer The pointer to free. If it is `nullptr`, nothing
		 *                    happens.
		 */
		static void freeMemoryUnregistered(void* const pointer);

		/**
		 * Sets, or unsets, the autofree flag.
		 *
		 * If the autofree flag is set, all reserved device memory is freed
		 * upon destruction of this instance, without raising warnings or
		 * errors.
		 *
		 * @param[in] enable Whether to enable the autofree feature.
		 */
		void setAutofree(const bool enable)
		{
			autofree = enable;
		}

		/**
		 * Returns whether the given pointer is a pointer to CUDA Device memory.
		 *
		 * Returns `false` if `ptr == nullptr`.
		 *
		 * @throw OpenMPCD::Exception Throws on an error.
		 *
		 * @param[in] ptr The pointer to check.
		 */
		static bool isDeviceMemoryPointer(const void* const ptr);

		/**
		 * Returns whether the given pointer is a pointer to Host memory.
		 *
		 * If this function returns true, it does necessarily not mean that it
		 * is permissible to dereference the pointer.
		 *
		 * Returns `false` if `ptr == nullptr`.
		 *
		 * @throw OpenMPCD::Exception Throws on an error.
		 *
		 * @param[in] ptr The pointer to check.
		 */
		static bool isHostMemoryPointer(const void* const ptr);

		/**
		 * Copies `count` elements of type `T` from the Host to the Device.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `src == nullptr`.
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `dest == nullptr`
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `src` is not a Host pointer.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `dest` is not a Device pointer.
		 *
		 * @tparam T The type of elements to copy.
		 *
		 * @param[in]  src   The source to copy from.
		 * @param[out] dest  The destination to copy to.
		 * @param[in]  count The number of elements of type `T` to copy.
		 */
		template<typename T>
			static void copyElementsFromHostToDevice(
				const T* const src, T* const dest, const std::size_t count);

		/**
		 * Copies `count` elements of type `T` from the Device to the Host.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `src == nullptr`.
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `dest == nullptr`
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `src` is not a Device pointer.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `dest` is not a Host pointer.
		 *
		 * @tparam T The type of elements to copy.
		 *
		 * @param[in]  src   The source to copy from.
		 * @param[out] dest  The destination to copy to.
		 * @param[in]  count The number of elements of type `T` to copy.
		 */
		template<typename T>
			static void copyElementsFromDeviceToHost(
				const T* const src, T* const dest, const std::size_t count);

		/**
		 * Copies `count` elements of type `T` from the Device to the Device.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `src == nullptr`.
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `dest == nullptr`
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `src` is not a Device pointer.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `dest` is not a Device pointer.
		 *
		 * @tparam T The type of elements to copy.
		 *
		 * @param[in]  src   The source to copy from.
		 * @param[out] dest  The destination to copy to.
		 * @param[in]  count The number of elements of type `T` to copy.
		 */
		template<typename T>
			static void copyElementsFromDeviceToDevice(
				const T* const src, T* const dest, const std::size_t count);

		/**
		 * Writes zero-bytes to the given Device memory region.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `start == nullptr`
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `start` is not a Device pointer.
		 *
		 * @tparam T The type of elements pointed to by `start`.
		 *
		 * @param[in] start
		 *            The first element to write zero-bytes to.
		 * @param[in] numberOfElements
		 *            The number of elements of type `T` to write zero-bytes to.
		 */
		template<typename T>
			static void zeroMemory(
				T* const start, const std::size_t numberOfElements);

		/**
		 * Returns whether the `count` elements of type `T` at `host` in Host
		 * memory have the same byte-wise representation as the `count` elements
		 * at `device` on the CUDA Device.
		 *
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `src == nullptr`.
		 * @throw OpenMPCD::NULLPointerException
		 *        If OPENMPCD_DEBUG is defined, throws if `dest == nullptr`
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `host` is not a Host pointer.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined,
		 *        throws if `device` is not a Device pointer.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If OPENMPCD_DEBUG is defined, throws if `count == 0`.
		 *
		 * @tparam T The type of elements to copy.
		 *
		 * @param[in]  host   Pointer to the data on the Host.
		 * @param[out] device Pointer to the data on the CUDA Device.
		 * @param[in]  count  The number of elements of type `T` to compare,
		 *                    which must not be `0`.
		 */
		template<typename T>
			static bool elementMemoryEqualOnHostAndDevice(
				const T* const host, const T* const device,
				const std::size_t count);

	private:
		/**
		 * Allocates Device memory for the given number of instances of the
		 * supplied type, and registers it with this instance.
		 * The returned pointer points to at least
		 * `instanceCount * sizeof(Pointee)` bytes of allocated Device memory.
		 *
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the buffer could not be allocated.
		 *
		 * @param[in] bufferSize The number of bytes to allocate at a minimum.
		 *                       If `0`, `nullptr` is returned.
		 */
		void* allocateMemoryInternal(const std::size_t bufferSize);

		/**
		 * Allocates Device memory for the given number of instances of the
		 * supplied type, without registering it with this instance.
		 * The returned pointer points to at least
		 * `instanceCount * sizeof(Pointee)` bytes of allocated Device memory.
		 *
		 * @throw OpenMPCD::MemoryManagementException
		 *        Throws if the buffer could not be allocated.
		 *
		 * @param[in] bufferSize The number of bytes to allocate at a minimum.
		 *                       If `0`, `nullptr` is returned.
		 */
		static
		void* allocateMemoryInternalUnregistered(const std::size_t bufferSize);

	private:
		/**
		 * The assignment operator.
		 */
		const DeviceMemoryManager operator=(const DeviceMemoryManager&);

	private:
		std::set<const void*> allocatedBuffers; ///< Holds all pointers that are currently allocated.
		bool autofree;                          ///< Whether to free all memory upon destruction of this instance.
}; //class DeviceMemoryManager
} //namespace CUDA
} //namespace OpenMPCD

#include <OpenMPCD/CUDA/ImplementationDetails/DeviceMemoryManager.hpp>

#endif
