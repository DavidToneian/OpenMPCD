/**
 * @file
 * Defines compiler-dependent macros.
 */

#ifndef OPENMPCD_UTILITY_COMPILERDETECTION_HPP
#define OPENMPCD_UTILITY_COMPILERDETECTION_HPP

#ifdef __NVCC__
	#define OPENMPCD_COMPILER_NVCC ///< Defined when compiling with NVIDIA NVCC.
#endif

#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	#define OPENMPCD_COMPILER_GCC ///< Defined when compiling with GNU GCC or G++
#endif

#endif //OPENMPCD_UTILITY_COMPILERDETECTION_HPP
