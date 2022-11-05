/**
 * @file
 * Defines platform-dependent macros.
 */

#ifndef OPENMPCD_UTILITY_PLATFORMDETECTION_HPP
#define OPENMPCD_UTILITY_PLATFORMDETECTION_HPP

#if defined(__linux__)
	#define OPENMPCD_PLATFORM_LINUX ///< Signifies the linux platform.
	#define OPENMPCD_PLATFORM_POSIX ///< Signifies a POSIX-compliant platform.
#elif defined(_WIN32)
	#define OPENMPCD_PLATFORM_WIN32 ///< Signifies the Win32 platform.
#endif

#ifdef __CUDACC__
	#define OPENMPCD_PLATFORM_CUDA ///< Signifies compilation of CUDA sources.
#endif

#endif //OPENMPCD_UTILITY_PLATFORMDETECTION_HPP
