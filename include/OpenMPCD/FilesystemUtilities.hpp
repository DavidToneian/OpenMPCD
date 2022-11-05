/**
 * @file
 * Defines the OpenMPCD::FilesystemUtilities class.
 */

#ifndef OPENMPCD_FILESYSTEMUTILITIES_HPP
#define OPENMPCD_FILESYSTEMUTILITIES_HPP

#include <string>

namespace OpenMPCD
{
/**
 * Provides utility functions for filesystem access.
 */
class FilesystemUtilities
{
	private:
		/**
		 * The constructor.
		 */
		FilesystemUtilities();

	public:
		/**
		 * Ensures that the given directory exists, creating it and its parents if necessary.
		 * @throw IOException Throws on failure.
		 * @param[in] path The directory's path.
		 */
		static void ensureDirectory(const std::string& path);

		/**
		 * Ensures that the parent directory of the path given exists.
		 * If the path given has no parent, nothing is done.
		 * @throw IOException Throws on failure.
		 * @param[in] path The path in question.
		 */
		static void ensureParentDirectory(const std::string& path);

}; //class FilesystemUtilities
} //namespace OpenMPCD

#endif
