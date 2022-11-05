/**
 * @file
 * Implementation details for the Vector2D class.
 */

#ifndef OPENMPCD_VECTOR2D_IMPLEMENTATION1_HPP
#define OPENMPCD_VECTOR2D_IMPLEMENTATION1_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

namespace OpenMPCD
{
template<typename T> class Vector2D;

/**
 * Namespace for implementation details for Vector2D.
 */
namespace Implementation_Vector2D
{

/**
 * Helper class to allow partial template specialization of
 * `OpenMPCD::Vector2D::dot`.
 *
 * @tparam T The underlying scalar type.
 */
template<typename T, typename=void>
class Dot
{
	private:
	Dot(); ///< The constructor.

	public:
	/**
	 * Returns the scalar product two vectors.
	 *
	 * The scalar product is defined such that the left-hand-side's components
	 * are complex-conjugated prior to multiplication with the right-hand-side's
	 * components.
	 *
	 * @param[in] lhs The left-hand-side.
	 * @param[in] rhs The right-hand-side.
	 */
	static
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const T dot(const Vector2D<T>& lhs, const Vector2D<T>& rhs);
}; //class Dot
} //namespace Implementation_Vector2D
} //namespace OpenMPCD

#endif //OPENMPCD_VECTOR2D_IMPLEMENTATION1_HPP
