/**
 * @file
 * Defines the `OPENMPCD_DEBUG_ASSERT` macro.
 */

#ifndef OPENMPCD_DEBUG_ASSERT_HPP
#define OPENMPCD_DEBUG_ASSERT_HPP

/**
 * @def OPENMPCD_DEBUG_ASSERT_HOST_EXCEPTIONTYPE
 * If `OPENMPCD_DEBUG` is defined, this macro checks that the given condition
 * evaluates to `true`, and throws an instance of `ExceptionType`
 * otherwise.
 *
 * If `OPENMPCD_DEBUG` is not defined, this macro does nothing.
 */

#ifdef OPENMPCD_DEBUG
	#include <OpenMPCD/Exceptions.hpp>
	#define OPENMPCD_DEBUG_ASSERT_HOST_EXCEPTIONTYPE(assertion, ExceptionType) \
		do { \
		if(!(assertion)) \
			OPENMPCD_THROW(ExceptionType, #assertion); \
		} while(false)
#else
	#define OPENMPCD_DEBUG_ASSERT_HOST_EXCEPTIONTYPE(assertion, ExceptionType) \
		do{} while(false)
#endif

/**
 * If `OPENMPCD_DEBUG` is defined, this macro checks that the given condition
 * evaluates to `true`, and throws an instance of `OpenMPCD::AssertionException`
 * otherwise.
 *
 * If `OPENMPCD_DEBUG` is not defined, this macro does nothing.
 */
#define OPENMPCD_DEBUG_ASSERT_HOST(assertion) \
	OPENMPCD_DEBUG_ASSERT_HOST_EXCEPTIONTYPE( \
		assertion, OpenMPCD::AssertionException)


/**
 * @def OPENMPCD_DEBUG_ASSERT_DEVICE
 * If `OPENMPCD_DEBUG` is defined, this macro checks that the given condition
 * evaluates to `true`, and calls `assert` otherwise.
 *
 * If `OPENMPCD_DEBUG` is not defined, this macro does nothing.
 */

#ifdef OPENMPCD_DEBUG
	#include <assert.h>
	#define OPENMPCD_DEBUG_ASSERT_DEVICE(assertion) \
		do { \
			assert(assertion); \
		} while(false)
#else
	#define OPENMPCD_DEBUG_ASSERT_DEVICE(assertion) do{} while(false)
#endif


/**
 * @def OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE
 *
 * Asserts that the given expression evaluates to `true`, but only if
 * `OPENMPCD_DEBUG` is defined.
 *
 * If CUDA `__device__` or `__global__` functions are being compiled, this macro
 * expands to `OPENMPCD_DEBUG_ASSERT_DEVICE`, and otherwise expands to
 * `OPENMPCD_DEBUG_ASSERT_HOST_EXCEPTIONTYPE`, with the supplied `ExceptionType`
 * being passed to that macro.
 */
#ifdef __CUDA_ARCH__
	#define OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(assertion, ExceptionType) \
		OPENMPCD_DEBUG_ASSERT_DEVICE(assertion)
#else
	#define OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(assertion, ExceptionType) \
		OPENMPCD_DEBUG_ASSERT_HOST_EXCEPTIONTYPE(assertion, ExceptionType)
#endif

/**
 * Asserts that the given expression evaluates to `true`, but only if
 * `OPENMPCD_DEBUG` is defined.
 *
 * If CUDA `__device__` or `__global__` functions are being compiled, this macro
 * expands to `OPENMPCD_DEBUG_ASSERT_DEVICE`, and otherwise expands to
 * `OPENMPCD_DEBUG_ASSERT_HOST`.
 */
#define OPENMPCD_DEBUG_ASSERT(assertion) \
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(assertion, OpenMPCD::AssertionException)

#endif //OPENMPCD_DEBUG_ASSERT_HPP
