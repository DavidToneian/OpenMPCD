/**
 * @file
 * Declares functionality in the `OpenMPCD::CUDA::DeviceCode::NormalMode`
 * namespace.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_NORMALMODE_HPP
#define OPENMPCD_CUDA_DEVICECODE_NORMALMODE_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

/**
 * Namespace for CUDA Device functionality related to normal modes.
 *
 * @see OpenMPCD::NormalMode
 */
namespace NormalMode
{

/**
 * Calculates a normal coordinate.
 *
 * @see OpenMPCD::NormalMode
 *
 * @tparam T The underlying floating-point data type.
 *
 * @param[in] i
 *            The index of the normal coordinate to compute, which must lie in
 *            the range `[0, N]`.
 * @param[in] vectors
 *            The input vectors to use. Must be a non-`nullptr` Device pointer
 *            pointing to at least `N` instances.
 * @param[in] N
 *            The number of input coordinate vectors, which must not be `0`.
 * @param[in] shift
 *            The shift parameter \f$ S \f$.
 */
template<typename T>
OPENMPCD_CUDA_DEVICE
const Vector3D<T> computeNormalCoordinate(
	const unsigned int i, const Vector3D<T>* const vectors,
	const std::size_t N, const T shift = 0);

/**
 * Calculates a normal coordinate.
 *
 * @see OpenMPCD::NormalMode
 *
 * @tparam T The underlying floating-point data type.
 *
 * @param[in] i
 *            The index of the normal coordinate to compute, which must lie in
 *            the range `[0, N]`.
 * @param[in] vectors
 *            The input vectors to use. Must be a non-`nullptr` Device pointer
 *            pointing to at least `3 * N` instances, which will be interpreted
 *            as `N` three-dimensional vectors, with the first three instances
 *            belonging to the first vector, and so forth.
 * @param[in] N
 *            The number of input coordinate vectors, which must not be `0`.
 * @param[in] shift
 *            The shift parameter \f$ S \f$.
 */
template<typename T>
OPENMPCD_CUDA_DEVICE
const Vector3D<T> computeNormalCoordinate(
	const unsigned int i, const T* const vectors, const std::size_t N,
	const T shift = 0);

/**
 * Calculates all normal coordinates for the given array of vectors.
 *
 * @see OpenMPCD::NormalMode
 *
 * @tparam T The underlying floating-point data type.
 *
 * @param[in]  vectors
 *             The input vectors to use. Must be a non-`nullptr` Device pointer
 *             pointing to at least `3 * N` instances, which will be interpreted
 *             as `N` three-dimensional vectors, with the first three instances
 *             belonging to the first vector, and so forth.
 * @param[in]  N
 *             The number of input coordinate vectors, which must not be `0`.
 * @param[out] result
 *             Buffer to store the results into. Must be a non-`nullptr` Device
 *             pointer pointing to at least `3 * (N + 1)` instances. The first
 *             three instances will be set to the three coordinates of the
 *             normal mode with index `0`, i.e. \f$ \vec{q}_0 \f$, and so forth
 *             for all normal modes up to and including \f$ \vec{q}_N \f$, where
 *             \f$ N \f$ is the function parameter `N`.
 * @param[in]  shift
 *             The shift parameter \f$ S \f$.
 */
template<typename T>
OPENMPCD_CUDA_DEVICE
void computeNormalCoordinates(
	const T* const vectors, const std::size_t N, T* const result,
	const T shift = 0);

/**
 * Calculates all normal coordinates for a number of polymer chains stored
 * contiguously in memory.
 *
 * @see OpenMPCD::NormalMode
 *
 * @tparam T The underlying floating-point data type.
 *
 * @param[in]  workUnitOffset
 *             The number of chains to skip.
 * @param[in]  chainLength
 *             The number of particles in a chain, which must be greater than
 *             `0`.
 * @param[in]  chainCount
 *             The number of chains present.
 * @param[in]  positions
 *             The Device array of particle positions.
 *             It is assumed that first, the `x`, `y`, and `z` coordinates of
 *             the first chain's first particle are stored, followed by the
 *             first chain's second particle, and so forth, up to the first
 *             chain's last particle. After that, the second chain's particles
 *             follow in a similar fashion.
 * @param[out] normalModeCoordinates
 *             A Device buffer that holds at least
 *             `3 * chainCount * (chainLength + 1)` elements, where the
 *             calculated normal mode coordinates will be saved.
 *             First, the normal mode coordinates of the first chain will be
 *             saved (starting with the `x`, `y`, and `z` coordinate of the
 *             normal mode `0`, followed by mode `1`, up to and including mode
 *             `N`), followed by the second chain, and so forth.
 * @param[in]  shift
 *             The shift parameter \f$ S \f$.
 */
template<typename T>
__global__ void computeNormalCoordinates(
	const unsigned int workUnitOffset,
	const unsigned int chainLength,
	const unsigned int chainCount,
	const T* const positions,
	T* const normalModeCoordinates,
	const T shift = 0);

} //namespace NormalMode
} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD


#include <OpenMPCD/CUDA/DeviceCode/ImplementationDetails/NormalMode.hpp>

#endif //OPENMPCD_CUDA_DEVICECODE_NORMALMODE_HPP
