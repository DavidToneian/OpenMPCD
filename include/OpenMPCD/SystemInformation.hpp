/**
 * @file
 * Defines the OpenMPCD::SystemInformation class.
 */

#ifndef OPENMPCD_SYSTEMINFORMATION_HPP
#define OPENMPCD_SYSTEMINFORMATION_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace OpenMPCD
{
/**
 * Provides information about the system the program is executing on.
 */
class SystemInformation
{
	public:
		/**
		 * Holds properties of a CUDA device.
		 */
		struct CUDADevice
		{
			std::string name;         ///< The device's name.
			std::size_t globalMemory; ///< The total amount of total memory available, in bytes.
		};

	private:
		SystemInformation(); ///< The constructor.

	public:
		/**
		 * Returns the hostname.
		 * @throw Exception Throws on errors.
		 */
		static const std::string getHostname();

		/**
		 * Returns the total number of bytes of physical main memory.
		 * @throw Exception Throws on errors.
		 */
		static std::size_t getTotalPhysicalMainMemory();

		/**
		 * Returns a list of cuda devices.
		 */
		static const std::vector<CUDADevice> getCUDADevices();
}; //class SystemInformation
} //namespace OpenMPCD

#endif
