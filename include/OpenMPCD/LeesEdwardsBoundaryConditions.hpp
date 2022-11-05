/**
 * @file
 * Defines Host functions for Lees-Edwards boundary conditions.
 */

#ifndef OPENMPCD_LEESEDWARDSBOUNDARYCONDITIONS_HPP
#define OPENMPCD_LEESEDWARDSBOUNDARYCONDITIONS_HPP

#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

/**
 * @page LeesEdwardsBoundaryConditions Lees-Edwards Boundary Conditions
 *
 * The Lees-Edwards boundary conditions are described in chapter 8.2 of
 * "Computer Simulation of Liquids" by M. P. Allen and D. J. Tildesley,
 * Oxford Science Publications,
 * as well as
 * "Multiparticle collision dynamics simulations of viscoelastic fluids:
 * Shear-thinning Gaussian dumbbells" by Bartosz Kowalik and Roland G. Winkler,
 * The Journal of Chemical Physics 138, 104903 (2013), DOI:10.1063/1.4792196.
 *
 * In this software, it is assumed that the primary simulation volume is
 * replicated along all three of the `x`, `y`, and `z` axes of the Cartesian
 * coordinate system. Images (including the primary simulation volume) that
 * are adjacent to one another along the `y` direction are moving such
 * that the image at larger `y` coordinate values has a relative
 * velocity along the `x` direction of \f$ \Delta v_x  = \dot{\gamma} L_y \f$
 * with respect to the other image at lower `y`, where \f$ \dot{\gamma} \f$ is
 * the shear rate, and \f$ L_y \f$ is the size of the primary simulation volume
 * along the `y` axis.
 */

namespace OpenMPCD
{
	/**
	 * Returns the image of the given particle position under Lees-Edwards
	 * boundary conditions.
	 *
	 * @see LeesEdwardsBoundaryConditions
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `velocityCorrection == nullptr`.
	 * @param[in]  position           The particle position.
	 * @param[in]  mpcTime            The simulation time for the MPC steps.
	 * @param[in]  shearRate          The applied shear rate.
	 * @param[in]  simBoxX            The size of the primary simulation box
	 *                                along the x direction.
	 * @param[in]  simBoxY            The size of the primary simulation box
	 *                                along the y direction.
	 * @param[in]  simBoxZ            The size of the primary simulation box
	 *                                along the z direction.
	 * @param[out] velocityCorrection Set to the velocity along the x direction
	 *                                that needs to be added to the particle's
	 *                                velocity.
	 */
	const Vector3D<MPCParticlePositionType>
		getImageUnderLeesEdwardsBoundaryConditions(
			const Vector3D<MPCParticlePositionType>& position,
			const FP mpcTime,
			const FP shearRate,
			const unsigned int simBoxX, const unsigned int simBoxY, const unsigned int simBoxZ,
			MPCParticleVelocityType* const velocityCorrection);

} //namespace OpenMPCD

#endif //OPENMPCD_LEESEDWARDSBOUNDARYCONDITIONS_HPP
