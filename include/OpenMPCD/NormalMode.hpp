/**
 * @file
 * Declares functionality in the `OpenMPCD::NormalMode` namespace.
 */

#ifndef OPENMPCD_NORMALMODE_HPP
#define OPENMPCD_NORMALMODE_HPP

#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{

/**
 * Namespace for functionality related to normal modes.
 *
 * Given a number \f$ N \f$ of coordinate vectors \f$ \vec{r}_n \f$,
 * \f$ n \in \left[ 1, N \right] \f$,
 * the \f$ i \f$-th normal coordinate \f$ \vec{q}_i \f$, with
 * \f$ i \in \left[ 0, N \right] \f$, is defined as
 * \f[
 * 		\vec{q}_i
 * 		=
 * 		\frac{1}{N} \sum_{n=1}^N
 * 		\cos\left( \frac{ i \pi \left( n + S \right) }{N} \right) \vec{r}_n
 * \f]
 * where \f$ S \f$ is a shift parameter.
 */
namespace NormalMode
{

/**
 * Calculates a normal coordinate.
 *
 * @see OpenMPCD::NormalMode
 *
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `vectors == nullptr`.
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if `N == 0`, or if `i` is out of
 *        range.
 *
 * @tparam T The underlying floating-point data type.
 *
 * @param[in] i
 *            The index of the normal coordinate to compute, which must lie in
 *            the range `[0, N]`.
 * @param[in] vectors
 *            The input vectors to use. Must be a non-`nullptr` pointer pointing
 *            to at least `N` instances.
 * @param[in] N
 *            The number of input coordinate vectors, which must not be `0`.
 * @param[in] shift
 *            The shift parameter \f$ S \f$.
 */
template<typename T>
const Vector3D<T> computeNormalCoordinate(
	const unsigned int i, const Vector3D<T>* const vectors,
	const std::size_t N, const T shift = 0);

} //namespace NormalMode
} //namespace OpenMPCD


#include <OpenMPCD/ImplementationDetails/NormalMode.hpp>

#endif //OPENMPCD_NORMALMODE_HPP
