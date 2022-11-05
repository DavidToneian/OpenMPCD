/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::GaussianRods class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_GAUSSIANRODS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_GAUSSIANRODS_HPP

#include <OpenMPCD/CUDA/MPCFluid/GaussianChains.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Base.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/OnTheFlyStatisticsDDDA.hpp>

#include <vector>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
/**
 * Instrumentation for Gaussian Chain fluids.
 *
 * If the `OpenMPCD::Configuration` contains a key
 * `instrumentation.gaussianChains.squaredBondLengths`, which must be set only
 * if there are at least two particles per chain, this object will measure the
 * squared lengths of each of the bonds, bond `0` being the one connecting
 * particles `0` and `1` in a chain, bond `1` connecting particles `1` and `2`,
 * and so forth up to bond `N_particlesPerChain - 1`.
 *
 * The data measured this way will be saved to the file
 * `gaussianChains--squaredBondLengths.txt`, which will contain (after comment
 * lines starting with `#`) a line for each bond, each consisting of
 *   - the bond number, starting at `0`,
 *   - the mean square bond length,
 *   - the sample size,
 *   - the standard deviation,
 *   - the optimal DDDA block ID, in the sense of
 *     `OpenMPCD::OnTheFlyStatisticsDDDA::getOptimalBlockIDForStandardErrorOfTheMean`,
 *   - the optimal standard error of the mean, in the sense of
 *     `OpenMPCD::OnTheFlyStatisticsDDDA::getOptimalStandardErrorOfTheMean`,
 *   - whether the standard error of the mean estimation is reliable, in the
 *     sense of
 *     `OpenMPCD::OnTheFlyStatisticsDDDA::optimalStandardErrorOfTheMeanEstimateIsReliable`,
 *     as either `1` for `true`, and `0` for `false`.
 */
class GaussianChains : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] chainLength The number of MPC fluid particles per chain.
		 * @param[in] sim         The simulation instance.
		 * @param[in] devMemMgr   The Device memory manager.
		 * @param[in] mpcFluid_   The MPC fluid to measure.
		 */
		GaussianChains(
			const unsigned int chainLength,
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::GaussianChains* const mpcFluid_);

		/**
		 * The destructor.
		 */
		virtual ~GaussianChains()
		{
			delete squaredBondLengths;
		}

	protected:
		/**
		 * Performs measurements.
		 */
		virtual void measureSpecific();

		/**
		 * Saves the data to the given run directory.
		 * @param[in] rundir The path to the run directory.
		 */
		virtual void saveSpecific(const std::string& rundir) const;

	private:
		const Simulation* const simulation; ///< The simulation instance.
		const MPCFluid::GaussianChains* const mpcFluid; ///< The fluid.

		std::vector<OnTheFlyStatisticsDDDA<MPCParticlePositionType> >*
			squaredBondLengths; ///< Measures the squares of the bond lengths.
}; //class GaussianChains
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
