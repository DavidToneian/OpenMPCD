/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::HarmonicTrimers class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_HARMONICTRIMERS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_HARMONICTRIMERS_HPP

#include <OpenMPCD/CUDA/MPCFluid/HarmonicTrimers.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Base.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Graph.hpp>
#include <OpenMPCD/Histogram.hpp>

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
 * Instrumentation for Harmonic Trimer fluids.
 */
class HarmonicTrimers : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim       The simulation instance.
		 * @param[in] devMemMgr The Device memory manager.
		 * @param[in] mpcFluid_ The MPC fluid to measure.
		 */
		HarmonicTrimers(
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::HarmonicTrimers* const mpcFluid_);

		/**
		 * The destructor.
		 */
		virtual ~HarmonicTrimers()
		{
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
		/**
		 * Returns the number of trimers.
		 */
		unsigned int getTrimerCount() const
		{
			return mpcFluid->getParticleCount()/3;
		}

		/**
		 * Returns the center of mass coordinates for the trimer with the given ID.
		 * @param[in] trimerID The ID of the trimer.
		 * @throw OutOfBoundsException If OPENMPCD_DEBUG is defined, throws if trimerID>=getTrimerCount()
		 */
		const Vector3D<MPCParticlePositionType> getTrimerCenterOfMass(const unsigned int trimerID) const;

	private:
		const Simulation* const simulation; ///< The simulation instance.
		const MPCFluid::HarmonicTrimers* const mpcFluid; ///< The fluid to measure.

		Histogram bond1LengthHistogram; ///< Histogram for the bond lengths between particles 1 and 2.
		Histogram bond2LengthHistogram; ///< Histogram for the bond lengths between particles 2 and 3.

		Histogram bond1LengthSquaredHistogram; ///< Histogram for the squares of the bond length between particles 1 and 2.
		Histogram bond2LengthSquaredHistogram; ///< Histogram for the squares of the bond length between particles 2 and 3.

		Histogram bond1XXHistogram; ///< Histogram for the squares of the x components of the bond 1.
		Histogram bond2XXHistogram; ///< Histogram for the squares of the x components of the bond 2.

		Histogram bond1YYHistogram; ///< Histogram for the squares of the y components of the bond 1.
		Histogram bond2YYHistogram; ///< Histogram for the squares of the y components of the bond 2.

		Histogram bond1ZZHistogram; ///< Histogram for the squares of the z components of the bond 1.
		Histogram bond2ZZHistogram; ///< Histogram for the squares of the z components of the bond 2.

		Histogram bond1XYHistogram; ///< Histogram for the product of the x and y components of the first bond.
		Histogram bond2XYHistogram; ///< Histogram for the product of the x and y components of the second bond.

		Histogram bond1XYAngleWithFlowDirectionHistogram; /**< Histogram for the angle between the 1-2 bond
		                                                       vector's projection to the xy plane and the flow
		                                                       direction.*/
		Histogram bond2XYAngleWithFlowDirectionHistogram; /**< Histogram for the angle between the 2-3 bond
		                                                       vector's projection to the xy plane and the flow
		                                                       direction.*/


		FP selfDiffusionCoefficientMeasurementTime;
			///< The time each measurement of the self-diffusion coefficient \f$D\f$ takes.
		Graph selfDiffusionCoefficients;
			///< Holds the measured self-diffusion coefficients \f$D\f$ and the time they ware taken.
		std::vector<Vector3D<MPCParticlePositionType> > trimerCenterOfMassesSnapshot;
			///< Holds the center-of-mass coordinates of the trimers at a given point in time.
		FP trimerCenterOfMassesSnapshotTime; ///< The simulation time at which trimerCenterOfMassesSnapshot was taken.
}; //class HarmonicTrimers
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
