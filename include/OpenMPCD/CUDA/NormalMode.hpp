/**
 * @file
 * Declares functionality in the `OpenMPCD::CUDA::NormalMode` namespace.
 */

#ifndef OPENMPCD_CUDA_NORMALMODE_HPP
#define OPENMPCD_CUDA_NORMALMODE_HPP

#include <OpenMPCD/Types.hpp>

#include <vector>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Namespace for Host-callable CUDA functionality related to normal modes.
 *
 * @see OpenMPCD::NormalMode
 */
namespace NormalMode
{

/**
 * Calculates all normal coordinates for a number of polymer chains stored
 * contiguously in Device memory.
 *
 * @see OpenMPCD::NormalMode
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if `chainLength == 0`.
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `normalModes0 == nullptr`.
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `normalModesT == nullptr`.
 *
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
 *             follow in a similar fashion, and so forth.
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
void computeNormalCoordinates(
	const unsigned int chainLength,
	const unsigned int chainCount,
	const MPCParticlePositionType* const positions,
	MPCParticlePositionType* const normalModeCoordinates,
	const MPCParticlePositionType shift = 0);

/**
 * Computes the average normal mode coordinate autocorrelation functions.
 *
 * With \f$ \vec{q}_i^k (t) \f$ being the normal mode coordinates for mode
 * \f$ i \f$ of chain \f$ k \f$ at time \f$ t \f$, this returns, for each
 * normal mode \f$ i \in \left[0, N\right] \f$ and with \f$ N_C \f$ being
 * the number of chains in the fluid, the average
 * \f$ N_C^{-1} \sum_{k=1}^{N_C} \vec{q}_i^k (0) \cdot \vec{q}_i^k (T) \f$,
 * where \f$ \cdot \f$ denotes the inner product.
 *
 * @see OpenMPCD::NormalMode
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        Throws if `chainLength == 0`.
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `normalModes0 == nullptr`.
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `normalModesT == nullptr`.
 *
 * @param[in]  chainLength    The number of particles in a chain, which must not
 *                            be `0`.
 * @param[in]  chainCount     The number \f$ N_C \f$ of chains.
 * @param[in]  normalModes0   Device pointer to the normal mode coordinates at
 *                            time \f$ 0 \f$, as calculated by
 *                            `calculateNormalModeCoordinatesForChain`.
 * @param[in]  normalModesT   Device pointer to the normal mode coordinates at
 *                            time \f$ T \f$, as calculated by
 *                            `calculateNormalModeCoordinatesForChain`.
 * @return
 * Returns the average autocorrelation of the normal mode vector
 * \f$ \vec{q}_i \f$ at index `i`, for all `i` in the range `[0, chainLength]`.
 */
const std::vector<MPCParticlePositionType>
	getAverageNormalCoordinateAutocorrelation(
		const unsigned int chainLength,
		const unsigned int chainCount,
		const MPCParticlePositionType* const normalModes0,
		const MPCParticlePositionType* const normalModesT);

} //namespace NormalMode
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_NORMALMODE_HPP
