/**
 * @file
 * Defines the `OpenMPCDTest::CUDA::SimulationTest` class.
 */

#ifndef OPENMPCDTEST_CUDA_SIMULATIONTEST_HPP
#define OPENMPCDTEST_CUDA_SIMULATIONTEST_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>

namespace OpenMPCDTest
{

/**
 * Namespace for tests of components in `OpenMPCD::CUDA`.
 */
namespace CUDA
{

/**
 * Helper to test `OpenMPCD::CUDA::Simulation`.
 */
class SimulationTest
{
	public:
		/**
		 * Tests `OpenMPCD::CUDA::Simulation::generateCollisionCellMBSFactors`.
		 */
		static void test_generateCollisionCellMBSFactors();

		/**
		 * Tests streaming of a simple fluid.
		 */
		static void test_stream_simpleFluid();

	public:
		/**
		 * Returns a `OpenMPCD::Configuration` instance for the tests.
		 */
		static const OpenMPCD::Configuration getConfiguration();

		/**
		 * Returns a `OpenMPCD::Configuration` instance for the tests which does
		 * not contain an MPC fluid.
		 */
		static const OpenMPCD::Configuration getConfigurationWithoutFluid();

		/**
		 * Returns a seed for the random number generator.
		 */
		static unsigned int getRNGSeed();

		/**
		 * Returns a pointer to the given instance's random number generator.
		 * @param[in] sim The simulation instance.
		 */
		static OpenMPCD::RNG* getRNG(OpenMPCD::CUDA::Simulation* const sim)
		{
			return &sim->rng;
		}
}; //class SimulationTest

} //namespace CUDA
} //namespace OpenMPCDTest

#endif //OPENMPCDTEST_CUDA_SIMULATIONTEST_HPP
