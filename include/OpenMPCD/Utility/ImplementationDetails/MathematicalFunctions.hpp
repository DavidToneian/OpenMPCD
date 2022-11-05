/**
 * @file
 * Implements common mathematical functions.
 */

#ifndef OPENMPCD_UTILITY_IMPLEMENTATIONDETAILS_MATHEMATICALFUNCTIONS_HPP
#define OPENMPCD_UTILITY_IMPLEMENTATIONDETAILS_MATHEMATICALFUNCTIONS_HPP

#include <OpenMPCD/Utility/MathematicalFunctions.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Utility/MathematicalConstants.hpp>
#include <OpenMPCD/Utility/PlatformDetection.hpp>

#include <cmath>

namespace OpenMPCD
{
namespace Utility
{
namespace MathematicalFunctions
{

///@cond

template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
float acos(const float x)
{
	return ::acosf(x);
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
double acos(const double x)
{
	return ::acos(x);
}
template<> OPENMPCD_CUDA_HOST inline
long double acos(const long double x)
{
	return ::acosl(x);
}



template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
float cos(const float x)
{
	return ::cosf(x);
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
double cos(const double x)
{
	return ::cos(x);
}
template<> OPENMPCD_CUDA_HOST inline
long double cos(const long double x)
{
	return ::cosl(x);
}



template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
float cospi(const float x)
{
	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::cospif(x);
	#else
		typedef float T;
		return MathematicalFunctions::cos(x * MathematicalConstants::pi<T>());
	#endif
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
double cospi(const double x)
{
	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::cospi(x);
	#else
		typedef double T;
		return MathematicalFunctions::cos(x * MathematicalConstants::pi<T>());
	#endif
}
template<> OPENMPCD_CUDA_HOST inline
long double cospi(const long double x)
{
	typedef long double T;
	return MathematicalFunctions::cos(x * MathematicalConstants::pi<T>());
}



template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
float sin(const float x)
{
	return ::sinf(x);
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
double sin(const double x)
{
	return ::sin(x);
}
template<> OPENMPCD_CUDA_HOST inline
long double sin(const long double x)
{
	return ::sinl(x);
}



template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
float sinpi(const float x)
{
	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::sinpif(x);
	#else
		typedef float T;
		return MathematicalFunctions::sin(x * MathematicalConstants::pi<T>());
	#endif
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
double sinpi(const double x)
{
	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::sinpi(x);
	#else
		typedef double T;
		return MathematicalFunctions::sin(x * MathematicalConstants::pi<T>());
	#endif
}
template<> OPENMPCD_CUDA_HOST inline
long double sinpi(const long double x)
{
	typedef long double T;
	return MathematicalFunctions::sin(x * MathematicalConstants::pi<T>());
}



template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
void sincos(const float x, float* const s, float* const c)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		s != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		c != NULL, OpenMPCD::NULLPointerException);

	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::sincosf(x, s, c);
	#else
		*s = MathematicalFunctions::sin(x);
		*c = MathematicalFunctions::cos(x);
	#endif
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
void sincos(const double x, double* const s, double* const c)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		s != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		c != NULL, OpenMPCD::NULLPointerException);

	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::sincos(x, s, c);
	#else
		*s = MathematicalFunctions::sin(x);
		*c = MathematicalFunctions::cos(x);
	#endif
}
template<> OPENMPCD_CUDA_HOST inline
void sincos(const long double x, long double* const s, long double* const c)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		s != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		c != NULL, OpenMPCD::NULLPointerException);

	*s = MathematicalFunctions::sin(x);
	*c = MathematicalFunctions::cos(x);
}


template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
void sincospi(const float x, float* const s, float* const c)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		s != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		c != NULL, OpenMPCD::NULLPointerException);

	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::sincospif(x, s, c);
	#else
		*s = MathematicalFunctions::sinpi(x);
		*c = MathematicalFunctions::cospi(x);
	#endif
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
void sincospi(const double x, double* const s, double* const c)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		s != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		c != NULL, OpenMPCD::NULLPointerException);

	#if defined(OPENMPCD_PLATFORM_CUDA) && defined(__CUDA_ARCH__)
		return ::sincospi(x, s, c);
	#else
		*s = MathematicalFunctions::sinpi(x);
		*c = MathematicalFunctions::cospi(x);
	#endif
}
template<> OPENMPCD_CUDA_HOST inline
void sincospi(const long double x, long double* const s, long double* const c)
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		s != NULL, OpenMPCD::NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		c != NULL, OpenMPCD::NULLPointerException);

	*s = MathematicalFunctions::sinpi(x);
	*c = MathematicalFunctions::cospi(x);
}



template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
float sqrt(const float x)
{
	return ::sqrtf(x);
}
template<> OPENMPCD_CUDA_HOST_AND_DEVICE inline
double sqrt(const double x)
{
	return ::sqrt(x);
}
template<> OPENMPCD_CUDA_HOST inline
long double sqrt(const long double x)
{
	return ::sqrtl(x);
}


///@endcond

} //namespace MathematicalFunctions
} //namespace Utility
} //namespace OpenMPCD

#endif //OPENMPCD_UTILITY_IMPLEMENTATIONDETAILS_MATHEMATICALFUNCTIONS_HPP
