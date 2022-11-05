/**
 * @file
 * Defines the OpenMPCD::RemotelyStoredVector class.
 */

#ifndef OPENMPCD_REMOTELYSTOREDVECTOR_HPP
#define OPENMPCD_REMOTELYSTOREDVECTOR_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <cmath>

namespace OpenMPCD
{
	/**
	 * Represents a vector whose data is stored elsewhere.
	 * @tparam T The type stored.
	 * @tparam D The dimensionality of the vector.
	 */
	template<typename T, unsigned int D=3> class RemotelyStoredVector
	{
		BOOST_STATIC_ASSERT(D != 0);

		public:
			/**
			 * The constructor.
			 * @param[in] storageBase The base address of the storage.
			 * @param[in] vectorID    The ID of the vector in the given storage.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			RemotelyStoredVector(T* const storageBase, const std::size_t vectorID=0)
				: storage(storageBase + D * vectorID)
			{
			}

			/**
			 * The copy constructor.
			 * @param[in] rhs The right-hand-side instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			RemotelyStoredVector(const RemotelyStoredVector<T, D>& rhs)
				: storage(rhs.storage)
			{
			}

		public:
			/**
			 * Returns the coodinate with the given index.
			 * @throw OutOfBoundsException If OPENMPCD_DEBUG is defined, in Host code, throws if i >= D.
			 * @param[in] i The coordinate index.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T get(const unsigned int i) const
			{
				#if defined(OPENMPCD_DEBUG) && !defined(__CUDA_ARCH__)
					if(i >= D)
						OPENMPCD_THROW(OutOfBoundsException, "i");
				#endif
				return storage[i];
			}

			/**
			 * Returns the x coodinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T getX() const
			{
				return storage[0];
			}

			/**
			 * Sets the x coordinate.
			 * @param[in] val The new value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void setX(const T val)
			{
				storage[0] = val;
			}

			/**
			 * Adds the given value to the x coordinate.
			 * @param[in] val The value to add
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void addToX(const T val)
			{
				storage[0] += val;
			}

			/**
			 * Returns the y coodinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T getY() const
			{
				BOOST_STATIC_ASSERT(D >= 2);

				return storage[1];
			}

			/**
			 * Sets the y coordinate.
			 * @param[in] val The new value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void setY(const FP val)
			{
				BOOST_STATIC_ASSERT(D >= 2);

				storage[1] = val;
			}

			/**
			 * Returns the z coodinate.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			FP getZ() const
			{
				BOOST_STATIC_ASSERT(D >= 3);

				return storage[2];
			}

			/**
			 * Sets the z coordinate.
			 * @param[in] val The new value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			void setZ(const FP val)
			{
				BOOST_STATIC_ASSERT(D >= 3);

				storage[2] = val;
			}

			/**
			 * Atomically adds the right-hand-side instance to this instance.
			 * @param[in] rhs The right-hand-side.
			 */
			OPENMPCD_CUDA_DEVICE
			void atomicAdd(const RemotelyStoredVector<typename boost::add_const<T>::type, D>& rhs);

			/**
			 * Atomically adds the right-hand-side instance to this instance.
			 * @param[in] rhs The right-hand-side.
			 */
			OPENMPCD_CUDA_DEVICE
			void atomicAdd(const RemotelyStoredVector<typename boost::remove_const<T>::type, D>& rhs);

			/**
			 * Atomically adds the right-hand-side instance to this instance.
			 * @param[in] rhs The right-hand-side.
			 */
			OPENMPCD_CUDA_DEVICE
			void atomicAdd(const Vector3D<typename boost::remove_const<T>::type>& rhs);

		public:
			/**
			 * Returns the scalar product of this vector with the given vector.
			 * @param[in] rhs The right-hand-side.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T dot(const RemotelyStoredVector& rhs) const
			{
				typename boost::remove_const<T>::type ret = 0;
				for(std::size_t i=0; i<D; ++i)
					ret += storage[i] * rhs.storage[i];
				return ret;
			}

			/**
			 * Returns the square of the magnitude of this vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T getMagnitudeSquared() const
			{
				return dot(*this);
			}

			/**
			 * Returns the magnitude of this vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			T getMagnitude() const
			{
				return sqrt(dot(*this));
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

				for(std::size_t i=0; i<D; ++i)
				{
					if(!isfinite(storage[i]))
						return false;
				}

				return true;
			}

		public:
			/**
			 * The assignment operator.
			 * @param[in] rhs The right-hand-side instance.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const RemotelyStoredVector& operator=(const RemotelyStoredVector& rhs)
			{
				for(std::size_t i=0; i<D; ++i)
					storage[i] = rhs.storage[i];

				return *this;
			}

			/**
			 * The assignment operator.
			 * @param[in] rhs The right-hand-side instance.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const RemotelyStoredVector& operator=(const Vector3D<T>& rhs)
			{
				BOOST_STATIC_ASSERT(D == 3);

				storage[0] = rhs.getX();
				storage[1] = rhs.getY();
				storage[2] = rhs.getZ();

				return *this;
			}

			/**
			 * The addition-and-assignment operator.
			 * @param[in] rhs The right-hand-side instance.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const RemotelyStoredVector& operator+=(const Vector3D<T>& rhs)
			{
				BOOST_STATIC_ASSERT(D == 3);

				storage[0] += rhs.getX();
				storage[1] += rhs.getY();
				storage[2] += rhs.getZ();

				return *this;
			}

			/**
			 * Addition operator.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D<typename boost::remove_const<T>::type>
				operator+(const RemotelyStoredVector& rhs) const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<typename boost::remove_const<T>::type>(
						getX() + rhs.getX(),
						getY() + rhs.getY(),
						getZ() + rhs.getZ());
			}

			/**
			 * Addition operator.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D<typename boost::remove_const<T>::type>
				operator+(const Vector3D<typename boost::remove_const<T>::type>& rhs) const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<T>(
						getX() + rhs.getX(),
						getY() + rhs.getY(),
						getZ() + rhs.getZ());
			}

			/**
			 * Substraction operator.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D<typename boost::remove_const<T>::type>
				operator-(const RemotelyStoredVector& rhs) const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<typename boost::remove_const<T>::type>(
						getX() - rhs.getX(),
						getY() - rhs.getY(),
						getZ() - rhs.getZ());
			}

			/**
			 * Substraction operator.
			 * @param[in] rhs The right-hand-side vector.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D<typename boost::remove_const<T>::type>
				operator-(const Vector3D<typename boost::remove_const<T>::type>& rhs) const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<T>(
						getX() - rhs.getX(),
						getY() - rhs.getY(),
						getZ() - rhs.getZ());
			}

			/**
			 * The scalar multiplication-and-assignment operator.
			 * @param[in] rhs The scalar factor.
			 * @return Returns this instance.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const RemotelyStoredVector& operator*=(const FP rhs)
			{
				BOOST_STATIC_ASSERT(D == 3);

				storage[0] *= rhs;
				storage[1] *= rhs;
				storage[2] *= rhs;

				return *this;
			}

			/**
			 * The scalar multiplication operator.
			 * @param[in] rhs The scalar factor.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D<typename boost::remove_const<T>::type> operator*(const FP rhs) const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<typename boost::remove_const<T>::type>(
						getX() * rhs,
						getY() * rhs,
						getZ() * rhs);
			}

			/**
			 * Scalar division operator.
			 * @param[in] rhs The right-hand-side scalar.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const Vector3D<typename boost::remove_const<T>::type> operator/(const FP rhs) const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<typename boost::remove_const<T>::type>(
						getX() / rhs,
						getY() / rhs,
						getZ() / rhs);
			}

			/**
			 * Scalar division-and-assignment operator.
			 * @param[in] rhs The right-hand-side scalar.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			const RemotelyStoredVector operator/=(const FP rhs)
			{
				BOOST_STATIC_ASSERT(D == 3);

				storage[0] /= rhs;
				storage[1] /= rhs;
				storage[2] /= rhs;

				return *this;
			}

			/**
			 * Converts this instance into a Vector3D.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			operator Vector3D<typename boost::remove_const<T>::type>() const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return Vector3D<typename boost::remove_const<T>::type>(getX(), getY(), getZ());
			}

			/**
			 * The comparison operator.
			 *
			 * @param[in] rhs The right-hand-side value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool operator==(
				const Vector3D<typename boost::remove_const<T>::type>& rhs)
			const
			{
				BOOST_STATIC_ASSERT(D == 3);

				if(storage[0] != rhs.getX())
					return false;
				if(storage[1] != rhs.getY())
					return false;
				if(storage[2] != rhs.getZ())
					return false;
				return true;
			}

			/**
			 * The inequality operator.
			 *
			 * @param[in] rhs The right-hand-side value.
			 */
			OPENMPCD_CUDA_HOST_AND_DEVICE
			bool operator!=(
				const Vector3D<typename boost::remove_const<T>::type>& rhs)
			const
			{
				BOOST_STATIC_ASSERT(D == 3);

				return !operator==(rhs);
			}

			/**
			 * Output operator for streams.
			 *
			 * Appends `vector.getX()`, `vector.getY()`, and `vector.getZ()` to
			 * the stream in that order, separated by single spaces (`" "`).
			 *
			 * @param[in] stream The stream to print to.
			 * @param[in] vector The vector to print.
			 */
			friend std::ostream&
			operator<<(std::ostream& stream, const RemotelyStoredVector& vector)
			{
				stream
					<< vector.getX() << " "
					<< vector.getY() << " "
					<< vector.getZ();
				return stream;
			}

		private:
			T* const storage; ///< The data storage.
	};



	/**
	 * Addition-and-assignment operator.
	 * @tparam L  The numeric type of the left-hand-side.
	 * @tparam R  The numeric type of the left-hand-side.
	 * @param[in] lhs The left-hand-side.
	 * @param[in] rhs The right-hand-side.
	 */
	template<typename L, typename R>
		OPENMPCD_CUDA_HOST_AND_DEVICE
		const Vector3D<L>& operator+=(Vector3D<L>& lhs, const RemotelyStoredVector<R, 3>& rhs)
	{
		lhs.addToX(rhs.getX());
		lhs.addToY(rhs.getY());
		lhs.addToZ(rhs.getZ());

		return lhs;
	}
}

#endif
