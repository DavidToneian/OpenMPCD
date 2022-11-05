/**
 * @file
 * Defines Device functions for Lees-Edwards boundary conditions.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_LEESEDWARDSBOUNDARYCONDITIONS_HPP
#define OPENMPCD_CUDA_DEVICECODE_LEESEDWARDSBOUNDARYCONDITIONS_HPP

#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{
	/**
	 * The relative `x`-velocity of `y`-adjacent layers in Lees-Edwards boundary
	 * conditions.
	 *
	 * @see LeesEdwardsBoundaryConditions
	 */
	extern __constant__ FP leesEdwardsRelativeLayerVelocity;


	/**
	 * Sets CUDA symbols for Lees-Edwards boundary conditions.
	 *
	 * @see LeesEdwardsBoundaryConditions
	 *
	 * This function stores \f$ \Delta v_x  = \dot{\gamma} L_y \f$ in
	 * `leesEdwardsRelativeLayerVelocity`.
	 *
	 * @param[in] shearRate The shear rate \f$ \dot{\gamma} \f$.
	 * @param[in] simBoxY   The size \f$ L_y \f$ of the primary simulation box
	 *                      along the y direction.
	 */
	void setLeesEdwardsSymbols(const FP shearRate, const unsigned int simBoxY);

	/**
	 * Returns the image of the given particle position under Lees-Edwards
	 * boundary conditions.
	 *
	 * This function assumes that `setLeesEdwardsSymbols` and
	 * `setSimulationBoxSizeSymbols` have been called before.
	 *
	 * @see LeesEdwardsBoundaryConditions
	 *
	 * @param[in]  mpcTime            The simulation time for the MPC steps.
	 * @param[in]  position           The particle position.
	 * @param[out] velocityCorrection Set to the velocity along the x direction that needs to
	 *                                be added to the particle's velocity.
	 */
	__device__ const Vector3D<MPCParticlePositionType>
		getImageUnderLeesEdwardsBoundaryConditions(
			const FP mpcTime,
			const Vector3D<MPCParticlePositionType>& position,
			MPCParticleVelocityType& velocityCorrection);

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif
