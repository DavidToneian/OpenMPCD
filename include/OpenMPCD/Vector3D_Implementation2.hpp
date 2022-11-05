/**
 * @file
 * Implementation details for the Vector3D class.
 */

#ifndef OPENMPCD_VECTOR3D_IMPLEMENTATION2_HPP
#define OPENMPCD_VECTOR3D_IMPLEMENTATION2_HPP

#include <OpenMPCD/Vector3D.hpp>

#include <boost/type_traits/is_floating_point.hpp>

namespace OpenMPCD
{
namespace Implementation_Vector3D
{

/**
 * Helper class to allow partial template specialization of
 * `OpenMPCD::Vector3D::dot`, for the case where `T` is a floating-point type.
 *
 * @see OpenMPCD::Implementation_Vector3D::Dot
 *
 * @tparam T The underlying scalar type.
 */
template<typename T>
class Dot<T, typename boost::enable_if<boost::is_floating_point<T> >::type>
{
	private:
	Dot();

	public:

	/**
	 * Returns the scalar product two real-valued vectors.
	 *
	 * @see OpenMPCD::Implementation_Vector3D::Dot::dot
	 *
	 * @param[in] lhs The left-hand-side.
	 * @param[in] rhs The right-hand-side.
	 */
	static
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const T dot(const Vector3D<T>& lhs, const Vector3D<T>& rhs)
	{
		BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

		return lhs.getX()*rhs.getX() + lhs.getY()*rhs.getY() + lhs.getZ()*rhs.getZ();
	}
}; //class Dot, partial template specialization for floating point types

/**
 * Helper class to allow partial template specialization of
 * `OpenMPCD::Vector3D::dot`, for the case where `T` is an instance of
 * `std::complex`.
 *
 * @see OpenMPCD::Implementation_Vector3D::Dot
 *
 * @tparam T The underlying scalar type.
 */
template<typename T>
class Dot<std::complex<T> >
{
	private:
	Dot();

	public:

	/**
	 * Returns the scalar product two vectors.
	 *
	 * The scalar product is defined such that the left-hand-side's components
	 * are complex-conjugated prior to multiplication with the right-hand-side's
	 * components.
	 *
	 * @see OpenMPCD::Implementation_Vector3D::Dot::dot
	 *
	 * @param[in] lhs The left-hand-side.
	 * @param[in] rhs The right-hand-side.
	 */
	static
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const std::complex<T> dot(
		const Vector3D<std::complex<T> >& lhs,
		const Vector3D<std::complex<T> >& rhs)
	{
		BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

		return
			std::conj(lhs.getX())*rhs.getX() +
			std::conj(lhs.getY())*rhs.getY() +
			std::conj(lhs.getZ())*rhs.getZ();
	}
}; //class Dot, partial template specialization for std::complex<T>
} //namespace Implementation_Vector3D
} //namespace OpenMPCD

#endif
