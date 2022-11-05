/**
 * @file
 * Defines functions that provide information about the host the program
 * is executed on.
 */

#ifndef OPENMPCD_UTILITY_HOSTINFORMATION_HPP
#define OPENMPCD_UTILITY_HOSTINFORMATION_HPP

#include <string>

namespace OpenMPCD
{

/**
 * Holds utility functionality.
 */
namespace Utility
{

/**
 * Provides functions that give information about the host the program is being
 * executed on.
 */
namespace HostInformation
{

/**
 * Returns the hostname of the machine this program is being executed on.
 */
const std::string getHostname();

/**
 * Returns the current time in UTC as a string, formatted as
 * `YYYY-MM-DDTHH:MM:SS`.
 *
 * @warning
 * This function may call `std::gmtime`, so its interal `std::tm` object may be
 * changed. Furthermore, this implies that this function might not be
 * thread-safe.
 */
const std::string getCurrentUTCTimeAsString();

} //namespace HostInformation
} //namespace Utility
} //namespace OpenMPCD

#endif //OPENMPCD_UTILITY_HOSTINFORMATION_HPP
