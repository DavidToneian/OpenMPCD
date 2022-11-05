/**
 * @file
 * Defines the `OpenMPCDTest::CUDA::SimulationTest` class.
 */

#ifndef OPENMPCDTEST_CUDA_MPCFLUID_BASETEST_HPP
#define OPENMPCDTEST_CUDA_MPCFLUID_BASETEST_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

#include <boost/shared_ptr.hpp>
#include <utility>

namespace OpenMPCDTest
{
namespace CUDA
{

/**
 * Namespace for tests of components in `OpenMPCD::CUDA::MPCFluid`.
 */
namespace MPCFluid
{

/**
 * Helper to test `OpenMPCD::CUDA::MPCFluid::Base`.
 */
class BaseTest : private OpenMPCD::CUDA::MPCFluid::Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim                The simulation instance.
		 * @param[in] count              The number of fluid particles.
		 * @param[in] streamingTimestep_ The timestep for a streaming step.
		 * @param[in] rng_               A random number generator to seed this instance's RNG with.
		 * @param[in] devMemMgr          The Device memory manager.
		 */
		BaseTest(
			const OpenMPCD::CUDA::Simulation* const sim, const unsigned int count, const OpenMPCD::FP streamingTimestep_,
			OpenMPCD::RNG& rng_, OpenMPCD::CUDA::DeviceMemoryManager* const devMemMgr);

	public:
		virtual unsigned int getNumberOfLogicalEntities() const
		{
			return getParticleCount() / particlesPerLogicalEntity;
		}

		virtual bool numberOfParticlesPerLogicalEntityIsConstant() const
		{
			return true;
		}

		virtual unsigned int getNumberOfParticlesPerLogicalEntity() const
		{
			return particlesPerLogicalEntity;
		}

	public:
		/**
		 * Tests `OpenMPCD::CUDA::MPCFluid::Base::getParticleCount`.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static void test_getParticleCount(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Tests invariants involving `fetchFromDevice` and `pushToDevice`.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static void testFetchAndPushInvariants(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Tests `setPositionsAndVelocities`.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static void test_setPositionsAndVelocities(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Tests `OpenMPCD::CUDA::MPCFluid::Base::writeToSnapshot`.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static void test_writeToSnapshot(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Tests `OpenMPCD::CUDA::MPCFluid::Base::findMatchingParticlesOnHost`.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static void test_findMatchingParticlesOnHost(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Tests
		 * `OpenMPCD::CUDA::MPCFluid::Base::saveLogicalEntityCentersOfMassToDeviceMemory`.
		 *
		 * @param[in] particlesPerCell
		 *            The number of particles per collision cell.
		 * @param[in] simulationBox
		 *            The number of collisoin cells in `x`, `y`, and `z`
		 *            direction, respectively.
		 */
		static void test_saveLogicalEntityCentersOfMassToDeviceMemory(
			const unsigned int particlesPerCell,
			const unsigned int (&simulationBox)[3]);

	public:
		/**
		 * Returns newly allocated simulation and MPC fluid instances.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static const std::pair<boost::shared_ptr<OpenMPCD::CUDA::Simulation>, boost::shared_ptr<BaseTest> >
			getSimulationAndFluid(const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Moves MPC particles on the Host, such that there are exactly `targetNumber` particles in the collision cell specified.
		 * This function does not take any boundary conditions into account. Since the particle positions
		 * can nominally have, e.g. in case of periodic or Lees-Edwards boundary conditions,
		 * coordinates outside of the primary simulation volume (which would then be temporarily
		 * mapped into the primary volume during collision),
		 * this function may yield unexpected results if any particles are outside the primary simulation volume.
		 * @param[in,out] fluid          The fluid to consider.
		 * @param[in]     targetNumber   The target number of MPC particles in the given collision cell.
		 * @param[in]     collisionCellX The `x` ID of the collision cell, starting at 0.
		 * @param[in]     collisionCellY The `y` ID of the collision cell, starting at 0.
		 * @param[in]     collisionCellY The `z` ID of the collision cell, starting at 0.
		 * @throw Exception Throws if the target cannot be achieved.
		 */
		static void achieveTargetNumberOfParticlesInCell_host(
			OpenMPCD::CUDA::MPCFluid::Base* const fluid,
			const unsigned int targetNumber,
			const unsigned int collisionCellX,
			const unsigned int collisionCellY,
			const unsigned int collisionCellZ);

		/**
		 * Returns the MPC timestep to use in tests.
		 */
		static OpenMPCD::FP getMPCTimestep()
		{
			return 0.1;
		}

		/**
		 * Returns a `OpenMPCD::Configuration` instance for the tests.
		 * @param[in] particlesPerCell The number of particles per collision cell.
		 * @param[in] simulationBox    The number of collisoin cells in `x`, `y`, and `z` direction, respectively.
		 */
		static const OpenMPCD::Configuration getConfiguration(
			const unsigned int particlesPerCell, const unsigned int (&simulationBox)[3]);

		/**
		 * Executes the given function with particle densities and simulation geometries selected from a list.
		 * @param[in] func The function to call, the first argument of which is the number of particles per
		 *                 collision cell, and the second argument being an array holding the number of
		 *                 collisoin cells in `x`, `y`, and `z` direction, respectively.
		 * @param[in] maxParticleCount
		 *                 The maximum number of particles to run tests for, or
		 *                 `0` for no limit.
		 */
		static void executeWithVariousDensitiesAndGeometries(
			void (*func)(const unsigned int, const unsigned int (&)[3]),
			const unsigned maxParticleCount = 0);

		/**
		 * Sets the number of particles per logical entity.
		 *
		 * @param[in] value
		 *            The value to set, which must divide `getParticleCount()`.
		 */
		void setParticlesPerLogicalentity(const unsigned int value);

	private:
		virtual void stream()
		{
		}

		unsigned int particlesPerLogicalEntity;
			///< The number of particles per logical entity.
}; //class BaseTest

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCDTest

#endif //OPENMPCDTEST_CUDA_MPCFLUID_BASETEST_HPP
