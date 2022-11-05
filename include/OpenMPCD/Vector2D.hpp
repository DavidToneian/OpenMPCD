/**
 * @file
 * Defines the `OpenMPCD::Vector2D` class template.
 */

#ifndef OPENMPCD_VECTOR2D_HPP
#define OPENMPCD_VECTOR2D_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Scalar.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/TypeTraits.hpp>
#include <OpenMPCD/Utility/MathematicalFunctions.hpp>
#include <OpenMPCD/Utility/PlatformDetection.hpp>
#include <OpenMPCD/Vector2D_Implementation1.hpp>

#include <boost/static_assert.hpp>
 #include <boost/type_traits/is_same.hpp>
 #include <boost/type_traits/remove_cv.hpp>

#include <cmath>
#include <complex>
#include <ostream>

namespace OpenMPCD
{

/**
 * 2-dimensional vector.
 *
 * @tparam T The underlying floating-point type.
 */
template<typename T> class Vector2D
{
public:
	typedef typename TypeTraits<T>::RealType RealType;
		///< The real-value type matching T.

public:
	/**
	 * Constructs a vector from the default values of the `T` type.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector2D() : x(), y()
	{
	}

	/**
	 * Constructs a vector from its coordinates.
	 *
	 * @param[in] x_ The x-coordinate.
	 * @param[in] y_ The y-coordinate.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector2D(const T x_, const T y_)
		: x(x_), y(y_)
	{
	}

public:
	/**
	 * Returns the x coordinate.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getX() const
	{
		return x;
	}

	/**
	 * Returns the y coordinate.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getY() const
	{
		return y;
	}

public:
	/**
	 * Returns the scalar product of this vector with the given vector.
	 *
	 * The scalar product is defines such that the left-hand-side's components
	 * are complex-conjugated prior to multiplication with the right-hand-side's
	 * components.
	 *
	 * @param[in] rhs The right-hand-side.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const T dot(const Vector2D& rhs) const
	{
		return Implementation_Vector2D::Dot<T>::dot(*this, rhs);
	}

	/**
	 * Returns the square of the magnitude of this vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	RealType getMagnitudeSquared() const
	{
		return Scalar::getRealPart(dot(*this));
	}

	/**
	 * Returns the magnitude of this vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	RealType getMagnitude() const
	{
		return
			OpenMPCD::Utility::MathematicalFunctions::sqrt(
				getMagnitudeSquared());
	}

	/**
	 * Returns the cosine of the angle between this vector and the given one.
	 *
	 * @tparam Result The result type.
	 *
	 * @param[in] rhs The right-hand-side vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getCosineOfAngle(const Vector2D& rhs) const
	{
		BOOST_STATIC_ASSERT(TypeTraits<T>::isStandardFloatingPoint);

		//This function tends to produce results that lie outside the
		//permissible range [-1, 1] when used with floats.
		BOOST_STATIC_ASSERT(
			!boost::is_same<
				float,
				typename boost::remove_cv<T>::type
			>::value);

		const T divisor =
			OpenMPCD::Utility::MathematicalFunctions::sqrt(
				getMagnitudeSquared() * rhs.getMagnitudeSquared());
		const T ret = dot(rhs) / divisor;

		OPENMPCD_DEBUG_ASSERT(-1 <= ret);
		OPENMPCD_DEBUG_ASSERT(ret <= 1);

		return ret;
	}

	/**
	 * Returns the the angle between this vector and the given one.
	 *
	 * @param[in] rhs The right-hand-side vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getAngle(const Vector2D& rhs) const
	{
		BOOST_STATIC_ASSERT(TypeTraits<T>::isStandardFloatingPoint);

		return
			OpenMPCD::Utility::MathematicalFunctions::acos(
				getCosineOfAngle(rhs));
	}

public:
	/**
	 * Equality operator.
	 *
	 * @param[in] rhs The right-hand-side vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	bool operator==(const Vector2D& rhs) const
	{
		return x == rhs.x && y == rhs.y;
	}

	/**
	 * Inequality operator.
	 *
	 * @param[in] rhs The right-hand-side vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	bool operator!=(const Vector2D& rhs) const
	{
		return !operator==(rhs);
	}

private:
	T x; ///< The x-coordinate.
	T y; ///< The y-coordinate.
}; //class Vector2D
} //namespace OpenMPCD

#include <OpenMPCD/Vector2D_Implementation2.hpp>

#endif //OPENMPCD_VECTOR2D_HPP
