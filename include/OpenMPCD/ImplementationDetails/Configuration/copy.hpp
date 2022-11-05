/**
 * @file
 * Defines the `OpenMPCD::ImplementationDetails::Configuration::copy` function.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_COPY_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_COPY_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Utility/PlatformDetection.hpp>

#include <cstdio>
#include <cstdlib>
#include <libconfig.h++>

namespace OpenMPCD
{
namespace ImplementationDetails
{
namespace Configuration
{

/**
 * Copies a `libconfig::Config` instance.
 * @param[in]  from The instance to copy from.
 * @param[out] to   The instance to copy to.
 */
inline void copy(const libconfig::Config& from, libconfig::Config* const to)
{
	#if defined(OPENMPCD_PLATFORM_POSIX)
		char* buffer;
		std::size_t bufferSize;

		FILE* const stream = open_memstream(&buffer, &bufferSize);

		if(!stream)
			OPENMPCD_THROW(Exception, "Failed to allocate stream.");
	#elif defined(OPENMPCD_PLATFORM_WIN32)
		char filename[L_tmpnam_s];
		if(tmpnam_s(filename) != 0)
			OPENMPCD_THROW(Exception, "Failed to create temporary file name.");

		FILE* stream;
		if(fopen_s(&stream, filename, "w+") != 0)
			OPENMPCD_THROW(Exception, "Failed to open temporary file.");
	#else
		#error Unknown platform
	#endif

	from.write(stream);

	rewind(stream);

	to->read(stream);

	fclose(stream);

	#ifdef OPENMPCD_PLATFORM_POSIX
		free(buffer);
	#endif
}

} //namespace Configuration
} //namespace ImplementationDetails
} //namespace OpenMPCD

#endif
