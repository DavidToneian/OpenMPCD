/**
 * @file
 * Defines the OpenMPCD::CUDA::Simulation class.
 */

#ifndef OPENMPCD_CUDA_SIMULATION_HPP
#define OPENMPCD_CUDA_SIMULATION_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/Base.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/CUDA/MPCSolute/Base.hpp>
#include <OpenMPCD/CUDA/Types.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>
#include <string>

namespace OpenMPCDTest
{
namespace CUDA
{
class SimulationTest;
} //namespace CUDA
} //namespace OpenMPCDTest


namespace OpenMPCD
{

/**
 * Namespace for simulations using CUDA.
 */
namespace CUDA
{
	/**
	 * MPCD simulation with Molecular Dynamics on CUDA-capable GPUs.
	 *
	 * For a description of the Maxwell-Boltzmann-Scaling Thermostat, see
	 * "Cell-level canonical sampling by velocity scaling for multiparticle
	 * collision dynamics simulations"
	 * by C. C. Huang, A. Chatterji, G. Sutmann, G. Gompper, and R. G. Winkler.
	 * Journal of Computational Physics 229 (2010) 168-177.
	 * DOI:10.1016/j.jcp.2009.09.024
	 */
	class Simulation
	{
		friend class OpenMPCDTest::CUDA::SimulationTest;

		public:
			/**
			 * The constructor.
			 * @param[in] configurationFilename The path to the simulation configuration.
			 * @param[in] rngSeed               The seed for the random number generator.
			 * @param[in] dir                   The directory where simulation runs will be saved.
			 */
			Simulation(const std::string& configurationFilename, const unsigned int rngSeed, const std::string& dir);

			/**
			 * The constructor.
			 * @param[in] configuration The configuration instance.
			 * @param[in] rngSeed       The seed for the random number generator.
			 */
			Simulation(const Configuration& configuration, const unsigned int rngSeed);

		private:
			Simulation(const Simulation&); ///< The copy constructor.

		public:
			/**
			 * The destructor.
			 */
			~Simulation();

		public:
			/**
			 * Returns the configuration.
			 */
			const Configuration& getConfiguration() const
			{
				return config;
			}

			/**
			 * Performs the warmup step.
			 */
			void warmup();

			/**
			 * Performs a sweep.
			 */
			void sweep();

			/**
			 * Returns the number of times `sweep` has completed, not counting
			 * warmup sweeps as performed by `warmup`.
			 */
			unsigned int getNumberOfCompletedSweeps() const;

			/**
			 * Returns the size of the primary simulation box along the x direction.
			 */
			unsigned int getSimulationBoxSizeX() const
			{
				return mpcSimulationBoxSizeX;
			}

			/**
			 * Returns the size of the primary simulation box along the y direction.
			 */
			unsigned int getSimulationBoxSizeY() const
			{
				return mpcSimulationBoxSizeY;
			}

			/**
			 * Returns the size of the primary simulation box along the z direction.
			 */
			unsigned int getSimulationBoxSizeZ() const
			{
				return mpcSimulationBoxSizeZ;
			}

			/**
			 * Returns the number of collision cells.
			 */
			unsigned int getCollisionCellCount() const
			{
				return
					getSimulationBoxSizeX() *
					getSimulationBoxSizeY() *
					getSimulationBoxSizeZ();
			}


			/**
			 * Returns whether an MPC fluid has been configured.
			 */
			bool hasMPCFluid() const
			{
				return mpcFluid != NULL;
			}

			/**
			 * Returns the MPC fluid.
			 *
			 * @throw OpenMPCD::InvalidCallException
			 *        If `OPENMPCD_DEBUG` is defined, throws if `!hasMPCFluid()`.
			 */
			const MPCFluid::Base& getMPCFluid() const
			{
				OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
					hasMPCFluid(), OpenMPCD::InvalidCallException);

				return *mpcFluid;
			}

			/**
			 * Returns the MPC solute
			 *
			 * @throw OpenMPCD::InvalidCallException
			 *        If `OPENMPCD_DEBUG` is defined, throws if no solute has
			 *        been configured.
			 */
			const MPCSolute::Base<
				MPCParticlePositionType, MPCParticleVelocityType>&
			getMPCSolute() const
			{
				#ifdef OPENMPCD_DEBUG
					if(!mpcSolute)
						OPENMPCD_THROW(InvalidCallException, "getMPCSolute");
				#endif

				return *mpcSolute;
			}

			/**
			 * Returns the boundary conditions.
			 */
			const BoundaryCondition::Base* getBoundaryConditions() const
			{
				return boundaryCondition;
			}

			/**
			 * Returns true if Simulation has an instance of
			 * MPCSolute::Base
			 */
			bool hasSolute() const
			{
				if (mpcSolute) return true;
				return false;
			}

			/**
			 * Returns the MPC time that has passed since the start of the simulation.
			 */
			FP getMPCTime() const
			{
				return mpcTime;
			}

			/**
			 * Returns the MPC timestep.
			 */
			FP getMPCTimestep() const
			{
				return mpcTimestep;
			}

			/**
			 * Returns the directory containing the Simulation run data.
			 */
			std::string getRundir() const
			{
				return rundir;
			}

			/**
			 * Returns the Device memory manager.
			 */
			DeviceMemoryManager* getDeviceMemoryManager()
			{
				return &deviceMemoryManager;
			}

		private:
			/**
			 * Reads the configuration.
			 * @throw std::runtime_error Throws if the configuration is invalid.
			 */
			void readConfiguration();

			/**
			 * Initializes the simulation.
			 */
			void initialize();

			/**
			 * Runs the simulation for the given number of steps.
			 * @param[in] stepCount The number of time steps to simulate.
			 */
			void run(const unsigned int stepCount);

			/**
			 * Performs a streaming step.
			 */
			void stream();

			/**
			 * Performs a SRD collision step.
			 */
			void collide();

			/**
			 * Generates a new grid shift vector and uploads it to the CUDA device.
			 */
			void generateGridShiftVector();

			/**
			 * Generates new rotation axes for the collision cells and uploads them to the CUDA device.
			 */
			void generateCollisionCellRotationAxes();

			/**
			 * Generates Maxwell-Boltzmann-Scaling factors for the collision
			 * cells and uploads them to the CUDA device.
			 *
			 * If the Maxwell-Boltzmann-Scaling thermostat is not configured,
			 * this function does nothing.
			 */
			void generateCollisionCellMBSFactors();

		private:
			const Simulation& operator=(const Simulation&); ///< The assignment operator.

		private:
			Configuration config; ///< The configuration for this simulation.

			unsigned int mpcSimulationBoxSizeX;    /**< The length of the MPC simulation box in the x direction,
			                                            in units of the collision cell size.*/
			unsigned int mpcSimulationBoxSizeY;    /**< The length of the MPC simulation box in the y direction,
			                                            in units of the collision cell size.*/
			unsigned int mpcSimulationBoxSizeZ;    /**< The length of the MPC simulation box in the z direction,
			                                            in units of the collision cell size.*/

			FP mpcTimestep;                        ///< The timestep for the MPC simulation.
			FP srdCollisionAngle;                  ///< The collision angle for the SRD collision step.
			FP gridShiftScale;                     ///< The factor the random grid shift vector is scaled with.

			FP bulkThermostatTargetkT;
				/**< The product of Boltzmann's constant and the bulk
				     thermostat's target temperature, or `0` if no bulk
				     thermostat is configured.*/

			unsigned int mpcSweepSize;             ///< The size of each MPC sweep.
			unsigned int numberOfCompletedSweeps;
				///< The number of sweeps that have been completed.


			MPCFluid::Base* mpcFluid; ///< The MPC fluid.

			MPCSolute::Base<MPCParticlePositionType, MPCParticleVelocityType>*
				mpcSolute; ///< The MPC solute.

			BoundaryCondition::Base* boundaryCondition;
				///< The boundary condition instance.

			const std::string rundir; ///< The directory where Simulation runs will be saved.

			mutable RNG rng; ///< The random number generator.
			CUDA::GPURNG* gpurngs; ///< Points to Device memory containing RNGs.


			FP mpcTime; ///< The MPC time that has passed since the start of the simulation.

			DeviceMemoryManager deviceMemoryManager; ///< The Device memory manager.

			MPCParticlePositionType* d_gridShift; ///< The grid shift vector on the Device.
			MPCParticleVelocityType* d_leesEdwardsVelocityShift; ///< Temporary Device variable to store velocity corrections due to Lees-Edwards boundary conditions.
			unsigned int* d_fluidCollisionCellIndices; ///< The collision cell indices for the fluid particles.
			unsigned int* d_collisionCellParticleCounts; ///< The number of particles in the collision cells.
			MPCParticleVelocityType* d_collisionCellMomenta; ///< The momenta in the collision cells.
			MPCParticlePositionType* collisionCellRotationAxes; ///< The rotation axes in the collision cells on the Host.
			MPCParticlePositionType* d_collisionCellRotationAxes; ///< The rotation axes in the collision cells on the Device.
			FP* collisionCellRelativeVelocityScalings; ///< The relative velocity scaling factors in the collision cells on the Host.
			FP* d_collisionCellRelativeVelocityScalings; ///< The relative velocity scaling factors in the collision cells on the Device.
			FP* d_collisionCellFrameInternalKineticEnergies;
				/**< Stores, for each collision cell, the sum of the kinetic
				     energies of the particles in that collision cell, as
				     measured in that collision cell's center-of-mass frame.*/
			FP* d_collisionCellMasses; ///< The masses of the collision cells.

			MPCParticleVelocityType* d_leesEdwardsVelocityShift_solute;
				/**< Temporary Device buffer to store Lees-Edwards velocity
				     shifts for solute particles.*/
			unsigned int* d_collisionCellIndices_solute;
				///< The collision cell indices for the solute particles.
	};
} //namespace OpenMPCD::CUDA
} //namespace OpenMPCD

#endif
