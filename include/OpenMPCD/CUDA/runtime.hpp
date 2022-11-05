/**
 * @file
 * Includes the CUDA runtime headers.
 */

#ifndef OPENMPCD_CUDA_RUNTIME_HPP
#define OPENMPCD_CUDA_RUNTIME_HPP

#if defined(__GNUC__) && (__GNUC__>4 || (__GNUC__==4 && __GNUC_MINOR__>=6))
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wlong-long"
#endif
#include <cuda_runtime.h>
#if defined(__GNUC__) && (__GNUC__>4 || (__GNUC__==4 && __GNUC_MINOR__>=6))
	#pragma GCC diagnostic pop
#endif

#endif
