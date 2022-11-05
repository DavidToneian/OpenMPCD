/**
 * @file
 * Defines the `OpenMPCD::CUDA::Device` class.
 */

#ifndef OPENMPCD_CUDA_DEVICE_HPP
#define OPENMPCD_CUDA_DEVICE_HPP

#include <OpenMPCD/CUDA/runtime.hpp>

#include <string>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Class representing a CUDA Device.
 */
class Device
{
	public:
		/**
		 * Represents the CUDA Device active at the moment of this instance's
		 * construction on the current thread.
		 */
		Device();

	private:
		/**
		 * The copy constructor.
		 */
		Device(const Device&);

	public:
		/**
		 * Returns the number of CUDA Devices available.
		 */
		static unsigned int getDeviceCount();

		/**
		 * Returns the PCI Domain ID of this Device.
		 */
		unsigned int getPCIDomainID() const;

		/**
		 * Returns the PCI Bus ID of this Device.
		 */
		unsigned int getPCIBusID() const;

		/**
		 * Returns the PCI Slot ID (also known as PCI Device ID) of this Device.
		 */
		unsigned int getPCISlotID() const;

		/**
		 * Returns a string representing the PCI address.
		 *
		 * The returned value is formatted as `DDDD:BB:SS`, where `DDDD` is the
		 * four-digit hexademical (lowercase) representation of the value
		 * returned by `getPCIDomainID`, `BB` is two hexademical (lowercase)
		 * digits corresponding to the return value of `getPCIBusID`, and `SS`
		 * likewise for `getPCISlotID`.
		 */
		const std::string getPCIAddressString() const;

		/**
		 * Returns the stack size, in bytes, per Device thread.
		 */
		std::size_t getStackSizePerThread() const;

		/**
		 * Sets the stack size, in bytes, per Device thread.
		 *
		 * @throw OpenMPCD::Exception
		 *        Throws if an error occurred.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        If `OPENMPCD_DEBUG` is defined, throws if `value == 0`.
		 *
		 * @param[in] value The new stack size, which must be positive.
		 */
		void setStackSizePerThread(const std::size_t value) const;

	private:
		/**
		 * The assignment operator.
		 */
		const Device& operator=(const Device&);

	private:
		int deviceID; ///< The CUDA Device ID of this instance.
		cudaDeviceProp properties; ///< The Device properties.
}; //class Device
} //namespace CUDA
} //namespace OpenMPCD

#endif
