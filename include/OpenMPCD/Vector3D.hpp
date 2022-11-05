/**
 * @file
 * Defines the Vector3D class.
 */

#ifndef OPENMPCD_VECTOR3D_HPP
#define OPENMPCD_VECTOR3D_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Scalar.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/TypeTraits.hpp>
#include <OpenMPCD/Utility/MathematicalFunctions.hpp>
#include <OpenMPCD/Utility/PlatformDetection.hpp>
#include <OpenMPCD/Vector3D_Implementation1.hpp>

#ifdef OPENMPCD_PLATFORM_CUDA
#include <OpenMPCD/CUDA/Types.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/Uniform0e1i.hpp>
#endif

#include <boost/math/constants/constants.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_floating_point.hpp>

#include <cmath>
#include <complex>
#include <ostream>

namespace OpenMPCD
{
	/**
	 * 3-dimensional vector.
	 * @tparam T The underlying floating-point type.
	 */
	template<typename T> class Vector3D
	{
		public:
			typedef typename TypeTraits<T>::RealType RealType; ///< The real-value type matching T.

		public:
			/**
			 * Constructs a vector from its coordinates.
			 * @param[in] x_ The x-coordinate.
			 * @param[in] y_ The y-coordinate.
			 * @param[in] z_ The z-coordinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			Vector3D(const T x_, const T y_, const T z_)
				: x(x_), y(y_), z(z_)
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
			 * Sets the x coordinate.
			 * @param[in] val The new value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void setX(const T val)
			{
				x=val;
			}

			/**
			 * Adds the given value to the x coordinate.
			 * @param[in] val The value to add
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void addToX(const T val)
			{
				x+=val;
			}

			/**
			 * Returns the y coordinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T getY() const
			{
				return y;
			}

			/**
			 * Sets the y coordinate.
			 * @param[in] val The new value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void setY(const T val)
			{
				y=val;
			}

			/**
			 * Adds the given value to the y coordinate.
			 * @param[in] val The value to add
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void addToY(const T val)
			{
				y+=val;
			}

			/**
			 * Returns the z coordinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T getZ() const
			{
				return z;
			}

			/**
			 * Sets the z coordinate.
			 * @param[in] val The new value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void setZ(const T val)
			{
				z=val;
			}

			/**
			 * Adds the given value to the z coordinate.
			 * @param[in] val The value to add
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void addToZ(const T val)
			{
				z+=val;
			}

			/**
			 * Sets the coordinates.
			 * @param[in] x_ The x coordinate.
			 * @param[in] y_ The y coordinate.
			 * @param[in] z_ The z coordinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void set(const T x_, const T y_, const T z_)
			{
				x = x_;
				y = y_;
				z = z_;
			}

		public:
			/**
			 * Returns the cross product of this vector with the given vector.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D cross(const Vector3D& rhs) const
			{
				const T cx=y*rhs.z-z*rhs.y;
				const T cy=z*rhs.x-x*rhs.z;
				const T cz=x*rhs.y-y*rhs.x;
				return Vector3D(cx, cy, cz);
			}

			/**
			 * Returns the scalar product of this vector with the given vector.
			 *
			 * The scalar product is defines such that the left-hand-side's components are complex-conjugated
			 * prior to multiplication with the right-hand-side's components.
		 	 *
			 * @param[in] rhs The right-hand-side.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const T dot(const Vector3D& rhs) const
			{
				return Implementation_Vector3D::Dot<T>::dot(*this, rhs);
			}

			/**
			 * Returns the cosine of the angle between this vector and the given one.
			 * @tparam Result The result type.
			 * @param[in] rhs The right-hand-side vector.
			 */
			T getCosineOfAngle(const Vector3D& rhs) const
			{
				return dot(rhs) / (magnitude() * rhs.magnitude());
			}

			/**
			 * Returns the the angle between this vector and the given one.
			 * @param[in] rhs The right-hand-side vector.
			 */
			T getAngle(const Vector3D& rhs) const
			{
				return acos(getCosineOfAngle(rhs));
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
			 * Returns the square of the magnitude of this vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			RealType magnitudeSquared() const
			{
				return getMagnitudeSquared();
			}

			/**
			 * Returns the magnitude of this vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			RealType getMagnitude() const
			{
				return sqrt(getMagnitudeSquared());
			}

			/**
			 * Returns the magnitude of this vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			RealType magnitude() const
			{
				return getMagnitude();
			}

			/**
			 * Normalizes this vector.
			 *
			 * @throw OpenMPCD::DivisionByZeroException
			 *        For Host code, throws if `magnitude() == 0`.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void normalize()
			{
				const RealType mag=magnitude();

				#ifndef __CUDA_ARCH__
					if(Scalar::isZero(mag))
						OPENMPCD_THROW(DivisionByZeroException, "");
				#endif

				operator/=(mag);
			}

			/**
			 * Returns this vector, but normalized.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D getNormalized() const
			{
				Vector3D tmp(*this);
				tmp.normalize();
				return tmp;
			}

			/**
			 * Returns the projection of this vector onto the given one.
			 * @param[in] onto The vector to project onto.
			 */
			const Vector3D getProjectedOnto(const Vector3D& onto) const
			{
				const Vector3D normalized = onto.getNormalized();
				return normalized.dot(*this) * normalized;
			}

			/**
			 * Returns the part of this vector that is perpendicular to the given vector.
			 * @param[in] rhs The vector the result should be perpendicular to.
			 */
			const Vector3D getPerpendicularTo(const Vector3D& rhs) const
			{
				return *this - getProjectedOnto(rhs);
			}

			/**
			 * Returns the complex version of this vector.
			 *
			 * This function only works if this vector has a real-value type.
			 */
			const Vector3D<std::complex<T> > getComplexVector() const
			{
				BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

				return Vector3D<std::complex<T> >(x, y, z);
			}

			/**
			 * Rotates this vector about the given axis by the given angle.
			 * @param[in] axis  The axis to rotate about, which is assumed to be normalized.
			 * @param[in] angle The angle to rotate with.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void rotateAroundNormalizedAxis(const Vector3D& axis, const T angle)
			{
				const T        thisDotAxis   = dot(axis);
				const Vector3D axisCrossThis = axis.cross(*this);
				const Vector3D projectionOntoAxis = axis*thisDotAxis;
				*this = projectionOntoAxis + cos(angle) * (*this - projectionOntoAxis) + sin(angle) * axisCrossThis;
			}

			/**
			 * Returns this vector, but rotated about the given axis by the given angle.
			 * @param[in] axis  The axis to rotate about, which is assumed to be normalized.
			 * @param[in] angle The angle to rotate with.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D getRotatedAroundNormalizedAxis(const Vector3D& axis, const T angle) const
			{
				Vector3D tmp(*this);
				tmp.rotateAroundNormalizedAxis(axis, angle);
				return tmp;
			}

			/**
			 * Returns whether at least one of the components of this vector is negative.
			 */
			bool hasNegativeComponent() const
			{
				return x<0 || y<0 || z<0;
			}

			/**
			 * Returns whether all components are finite, i.e. neither infinite nor NaN.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool isFinite() const
			{
				#ifndef __CUDA_ARCH__
					using std::isfinite;
				#endif

				if(!isfinite(x))
					return false;
				if(!isfinite(y))
					return false;
				if(!isfinite(z))
					return false;

				return true;
			}

		public:
			/**
			 * Returns a random vector with unit length;
			 * all directions are equally likely.
			 *
			 * @tparam    RNG The random-number-generator type.
			 *
			 * @param[in] rng The random-number-generator.
			 */
			template<typename RNG> static OPENMPCD_CUDA_HOST
			const Vector3D getRandomUnitVector(RNG& rng)
			{
				boost::random::uniform_01<T> dist0i1e;

				//Ideally, X_1 should be sampled from the uniform distribution
				//over [0,1]
				const T X_1 = dist0i1e(rng);
				const T X_2 = dist0i1e(rng);

				return getUnitVectorFromRandom01(X_1, X_2);
			}

			#ifdef OPENMPCD_PLATFORM_CUDA
			static OPENMPCD_CUDA_DEVICE
			const Vector3D getRandomUnitVector(CUDA::GPURNG& rng)
			{
				CUDA::Random::Distributions::Uniform0e1i<T> dist0e1i;

				//Ideally, X_1 should be sampled from the uniform distribution
				//over [0,1]
				T X_1;
				T X_2;
				dist0e1i(rng, &X_1, &X_2);

				return getUnitVectorFromRandom01(X_1, X_2);
			}
			#endif

			/**
			 * Constructs a unit vector from the two given random variables.
			 *
			 * @param[in] X_1
			 *            A variable drawn from the uniform distribution on the
			 *            closed interval \f$ \left[ 0, 1 \right] \f$.
			 * @param[in] X_2
			 *            A variable drawn from the uniform distribution on
			 *            either the half-open interval
			 *            \f$ \left[ 0, 1 \right) \f$
			 *            or the half-open interval \f$ \left( 0, 1 \right] \f$.
			 */
			static OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D getUnitVectorFromRandom01(const T X_1, const T X_2)
			{
				namespace MF = OpenMPCD::Utility::MathematicalFunctions;

				const T z         = 2 * X_1 - 1;
				const T phiOverPi =	2 * X_2;

				const T root = MF::sqrt(1 - z*z);

				T x;
				T y;
				MF::sincospi(phiOverPi, &y, &x);

				x *= root;
				y *= root;

				return Vector3D(x, y, z);
			}

		public:
			/**
			 * Equality operator.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool operator==(const Vector3D& rhs) const
			{
				if(x == rhs.x && y == rhs.y && z == rhs.z)
					return true;
				return false;
			}

			/**
			 * Inequality operator.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool operator!=(const Vector3D& rhs) const
			{
				return !operator==(rhs);
			}

			/**
			 * Addition-and-assignment operator.
			 * @param[in] rhs The right-hand-side vector.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D& operator+=(const Vector3D& rhs)
			{
				x+=rhs.x;
				y+=rhs.y;
				z+=rhs.z;

				return *this;
			}

			/**
			 * Addition operator.
			 * @param[in] rhs The right-hand-side vector.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D operator+(const Vector3D& rhs) const
			{
				Vector3D tmp(*this);
				return tmp+=rhs;
			}

			/**
			 * Subtraction-and-assignment operator.
			 * @param[in] rhs The right-hand-side vector.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D& operator-=(const Vector3D& rhs)
			{
				x-=rhs.x;
				y-=rhs.y;
				z-=rhs.z;

				return *this;
			}

			/**
			 * Subtraction operator.
			 * @param[in] rhs The right-hand-side vector.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D operator-(const Vector3D& rhs) const
			{
				Vector3D tmp(*this);
				return tmp-=rhs;
			}

			/**
			 * The negation operator.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D operator-() const
			{
				return operator*(-1);
			}

			/**
			 * Scalar multiplication and assignment operator.
			 * @param[in] scale The scaling factor.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D& operator*=(const T scale)
			{
				x*=scale;
				y*=scale;
				z*=scale;

				return *this;
			}

			/**
			 * Scalar multiplication operator.
			 * @param[in] scale The scaling factor.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D operator*(const T scale) const
			{
				Vector3D tmp(*this);
				return tmp*=scale;
			}

			/**
			 * Scalar multiplication operator.
			 * @param[in] scale The scaling factor.
			 * @param[in] vec   The vector to multiply.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			friend const Vector3D operator*(const T scale, const Vector3D& vec)
			{
				return vec*scale;
			}

			/**
			 * Scalar division and assignment operator.
			 * @throw OpenMPCD::DivisionByZeroException
			 *        For Host code, throws if divisor is 0.
			 * @param[in] divisor The divisor.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D& operator/=(const T divisor)
			{
				#ifndef __CUDA_ARCH__
					if(Scalar::isZero(divisor))
						OPENMPCD_THROW(DivisionByZeroException, "");
				#endif

				x/=divisor;
				y/=divisor;
				z/=divisor;

				return *this;
			}

			/**
			 * Scalar division operator.
			 * @throw DivisionByZeroExcpetion For Host code, throws if divisor is 0.
			 * @param[in] divisor The divisor.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D operator/(const T divisor) const
			{
				Vector3D tmp(*this);
				return tmp/=divisor;
			}

			/**
			 * Less-than operator.
			 * Compares, in this order, the x, y, and z components.
			 * @param[in] rhs The right-hand-side instance.
			 */
			bool operator<(const Vector3D& rhs) const
			{
				if(x < rhs.x)
					return true;

				if(x > rhs.x)
					return false;

				if(y < rhs.y)
					return true;

				if(y > rhs.y)
					return false;

				if(z < rhs.z)
					return true;

				return false;
			}

			/**
			 * Output operator for streams.
			 * @param[in] stream The stream to print to.
			 * @param[in] vector The vector to print.
			 */
			friend std::ostream& operator<<(std::ostream& stream, const Vector3D& vector)
			{
				stream<<vector.x<<" "<<vector.y<<" "<<vector.z;
				return stream;
			}

		private:
			T x; ///< The x-coordinate.
			T y; ///< The y-coordinate.
			T z; ///< The z-coordinate.
	};
}

#include <OpenMPCD/Vector3D_Implementation2.hpp>

#endif
