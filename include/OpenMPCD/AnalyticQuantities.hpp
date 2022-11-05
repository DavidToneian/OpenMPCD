/**
 * @file
 * Defines the OpenMPCD::AnalyticQuantities class.
 */

#ifndef OPENMPCD_ANALYTICQUANTITIES_HPP
#define OPENMPCD_ANALYTICQUANTITIES_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
	/**
	 * Computes analytic quantities.
	 */
	class AnalyticQuantities
	{
		private:
			AnalyticQuantities(); ///< The constructor.

		public:
			/**
			 * Returns the analytical value for the mean free path for a point particle.
			 * @param[in] kT       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] m        The fluid particle mass.
			 * @param[in] timestep The timestep used in the simulation.
			 */
			static FP meanFreePath(const FP kT, const FP m, const FP timestep);

			/**
			 * Returns the kinetic contributions to the kinematic shear viscosity in SRD.
			 * This quantity is often denoted \f$ \nu^k \f$.
			 * The formula used is given in "Multi-Particle Collision Dynamics: A Particle-Based Mesoscale Simulation
			 * Approach to the Hydrodynamics of Complex Fluids" by G. Gompper, T. Ihle, D.M. Kroll, and R.G. Winkler,
			 * DOI:10.1007/978-3-540-87706-6_1, chapter 4.1.1, formula 32.
			 * @param[in] kT                       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] m                        The fluid particle mass.
			 * @param[in] meanParticleCountPerCell The mean number of particles per collision cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP kineticContributionsToSRDKinematicShearViscosity(
					const FP kT, const FP m, const FP meanParticleCountPerCell, const FP srdAngle, const FP timestep);

			/**
			 * Returns the kinetic contributions to the dynamic shear viscosity in SRD.
			 * This quantity is often denoted \f$ \eta^k \f$.
			 * The computed value is based on kineticContributionsToSRDKinematicShearViscosity.
			 * @param[in] kT                       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] m                        The fluid particle mass.
			 * @param[in] linearCellSize           The size of each of the cubic collision cell's edges.
			 * @param[in] meanParticleCountPerCell The mean number of particles per collision cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP kineticContributionsToSRDDynamicShearViscosity(
					const FP kT, const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,
					const FP srdAngle, const FP timestep);

			/**
			 * Returns the collisional contributions to the kinematic shear viscosity in SRD.
			 * This quantity is often denoted \f$ \nu^c \f$.
			 * The formula used is given in "Multi-Particle Collision Dynamics: A Particle-Based Mesoscale Simulation
			 * Approach to the Hydrodynamics of Complex Fluids" by G. Gompper, T. Ihle, D.M. Kroll, and R.G. Winkler,
			 * DOI:10.1007/978-3-540-87706-6_1, chapter 4.1.1, formula 39.
			 * @param[in] linearCellSize           The size of each of the cubic collision cell's edges.
			 * @param[in] meanParticleCountPerCell The mean number of particles per collision cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP collisionalContributionsToSRDKinematicShearViscosity(
					const FP linearCellSize, const FP meanParticleCountPerCell, const FP srdAngle,
					const FP timestep);

			/**
			 * Returns the collisional contributions to the dynamic shear viscosity in SRD.
			 * This quantity is often denoted \f$ \eta^c \f$.
			 * The computed value is based on collisionalContributionsToSRDKinematicShearViscosity.
			 * @param[in] m                        The fluid particle mass.
			 * @param[in] linearCellSize           The size of each of the cubic collision cell's edges.
			 * @param[in] meanParticleCountPerCell The mean number of particles per collision cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP collisionalContributionsToSRDDynamicShearViscosity(
					const FP m, const FP linearCellSize, const FP meanParticleCountPerCell, const FP srdAngle,
					const FP timestep);

			/**
			 * Returns the kinematic shear viscosity in SRD.
			 * This quantity is often denoted \f$ \nu \f$.
			 * The computed value is the sum of kineticContributionsToSRDKinematicShearViscosity
			 * and collisionalContributionsToSRDKinematicShearViscosity
			 * @param[in] kT                       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] m                        The fluid particle mass.
			 * @param[in] linearCellSize           The size of each of the cubic collision cell's edges.
			 * @param[in] meanParticleCountPerCell The mean number of particles per collision cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP SRDKinematicShearViscosity(
					const FP kT, const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,
					const FP srdAngle, const FP timestep);

			/**
			 * Returns the dynamic shear viscosity in SRD.
			 * This quantity is often denoted \f$ \eta \f$.
			 * The computed value is the sum of kineticContributionsToSRDDynamicShearViscosity
			 * and collisionalContributionsToSRDDynamicShearViscosity.
			 * @param[in] kT                       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] m                        The fluid particle mass.
			 * @param[in] linearCellSize           The size of each of the cubic collision cell's edges.
			 * @param[in] meanParticleCountPerCell The mean number of particles per collision cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP SRDDynamicShearViscosity(
					const FP kT, const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,
					const FP srdAngle, const FP timestep);

			/**
			 * Returns an approximation for the self-diffusion coefficient $D$.
			 * As given in Table 1 of "Multi-Particle Collision Dynamics: A Particle-Based Mesoscale Simulation
			 * Approach to the Hydrodynamics of Complex Fluids" by G. Gompper, T. Ihle, D.M. Kroll, and R.G. Winkler,
			 * DOI:10.1007/978-3-540-87706-6_1
			 * @param[in] dimensions The spatial dimensions of the system.
			 * @param[in] kT                       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] m                        The fluid particle mass.
			 * @param[in] meanParticleCountPerCell The average number of particles in a simulation cell.
			 * @param[in] srdAngle                 The angle used in SRD rotations, in radians.
			 * @param[in] timestep                 The timestep used in the simulation.
			 */
			static FP approximateSelfDiffusionCoefficient(const unsigned int dimensions, const FP kT, const FP m,
			                                              const FP meanParticleCountPerCell, const FP srdAngle,
			                                              const FP timestep);

			/**
			 * Returns the hydrodynamic radius \f$ R_H \f$ (Stokes-Einstein radius) of a spherical particle.
			 * @param[in] kT                       The product of the fluid temperature with Boltzmann's constant.
			 * @param[in] dynamicStressViscosity   The dynamic stress viscosity (\f$ \eta \f$ or \f$ \mu \f$) of the fluid.
			 * @param[in] selfDiffusionCoefficient The self-diffusion coefficient $D$ of the particle.
			 */
			static FP hydrodynamicRadius(const FP kT, const FP dynamicStressViscosity,
			                             const FP selfDiffusionCoefficient);
	};
}

#endif
