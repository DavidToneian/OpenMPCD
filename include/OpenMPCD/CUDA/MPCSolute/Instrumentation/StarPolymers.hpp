/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCSolute::Instrumentation::StarPolymers class.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_INSTRUMENTATION_STARPOLYMERS_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_INSTRUMENTATION_STARPOLYMERS_HPP

#include <OpenMPCD/CUDA/MPCSolute/Instrumentation/Base.hpp>
#include <OpenMPCD/CUDA/MPCSolute/StarPolymers.hpp>
#include <OpenMPCD/VTFSnapshotFile.hpp>

#include <boost/scoped_ptr.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{
namespace Instrumentation
{

/**
 * Instrumentation for `OpenMPCD::CUDA::MPCSolute::StarPolymers`.
 *
 * @tparam PositionCoordinate The type to store position coordinates.
 * @tparam VelocityCoordinate The type to store velocity coordinates.
 */
template<
	typename PositionCoordinate,
	typename VelocityCoordinate>
class StarPolymers : public Base
{
public:
	/**
	 * The constructor.
	 *
	 * @param[in] starPolymers_ The solute instance.
	 * @param[in] settings_     The configuration settings that have been used
	 *                          to configure the given star polymers instance.
	 */
	StarPolymers(
		MPCSolute::StarPolymers<PositionCoordinate, VelocityCoordinate>* const
			starPolymers_,
		const Configuration::Setting& settings_);

	/**
	 * The destructor.
	 */
	virtual ~StarPolymers();

protected:
	virtual void measureSpecific();
	virtual void saveSpecific(const std::string& rundir) const;

private:
	/**
	 * Resets the snapshot file.
	 */
	void resetSnapshotFile() const;

private:
	MPCSolute::StarPolymers<PositionCoordinate, VelocityCoordinate>* const
		starPolymers; ///< The solute instance.
	const Configuration::Setting settings;
		///< The configuration settings for the star polymers instance.

	mutable std::string snapshotFilePath; ///< The path to the snapshot file.
	mutable boost::scoped_ptr<VTFSnapshotFile> snapshotFile;
		///< The file storing simulation snapshots.
}; //class StarPolymers

} //namespace Instrumentation
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_MPCSOLUTE_INSTRUMENTATION_STARPOLYMERS_HPP
