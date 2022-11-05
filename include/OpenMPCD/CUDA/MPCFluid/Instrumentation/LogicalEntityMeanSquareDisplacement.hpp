/**
 * @file
 * Defines the
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::LogicalEntityMeanSquareDisplacement`
 * class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_LOGICALENTITYMEANSQUAREDISPLACEMENT_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_LOGICALENTITYMEANSQUAREDISPLACEMENT_HPP


#include <OpenMPCD/CUDA/DeviceBuffer.hpp>
#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/Types.hpp>

#include <ostream>
#include <vector>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{

class Base;

namespace Instrumentation
{

/**
 * Measures the mean square displacement of logical entities in an MPC fluid.
 *
 * This class can only be used if
 * `OpenMPCD::CUDA::MPCFluid::Base::numberOfParticlesPerLogicalEntityIsConstant`
 * is true.
 *
 * Let \f$ \vec{R}_i \f$ be the center of mass for the logical entity \f$ i \f$,
 * with \f$ i \in \left[0, N_L - 1 \right]\f$, where \f$ N_L \f$ is the return
 * value of `OpenMPCD::CUDA::MPCFluid::Base::getNumberOfLogicalEntities`.
 *
 * Then, this class samples, for configurable times \f$ \Delta t \f$,
 * the quantity
 * \f[
 * 		C \left( t, t + \Delta t \right)
 * 		=
 * 		\frac{1}{N_L}
 * 		\sum_{i=1}^{N_L}
 * 		\left(
 * 			\vec{R}_i \left( t + \Delta t \right)
 * 			-
 * 			\vec{R}_i \left( t \right)
 * 		\right)^2
 * \f]
 *
 * In order to avoid floating-point arithmetic in specifying times, and in order
 * to decouple this class from the `CUDA::Simulation` class, time is measured
 * in this class as the number of times `measure` has completed, not counting
 * those calls that have no effect because of the `measureEveryNthSweep`
 * configuration option (see below).
 * As an example, take `measureEveryNthSweep` to be `3`. Then, the first
 * execution of `measure` will perform a measurement, and that point in
 * simulation time will be referred to as `measurement time 0`. The next two
 * calls to `measure` will again have no effect, while the following (i.e. the
 * fourth) call will be at `measurement time 1`. The fifth and sixth calls
 * will again have no effect, and so forth.
 *
 * This class is configured via the
 * `instrumentation.logicalEntityMeanSquareDisplacement`
 * configuration group. Within that group, `measureEveryNthSweep` defines
 * \f$ \tau \f$, and `measurementArgumentCount` defines \f$ N_A \f$; given
 * those, the mean square displacement is measured at (simulation) times
 * \f[
 * 		\Delta t \in
 * 		\left\{
 * 			\tau \Delta T, 2 \tau \Delta T, \ldots, N_A \tau \Delta T
 * 		\right\}
 * \f]
 * where \f$ \Delta T \f$ is the simulation time between consecutive sweeps.
 * Consequently, in `measurement time`,
 * \f$ \Delta t \in \left\{ 1, 2, \ldots, N_A \right\} \f$.
 *
 * For analysis of the produced results, see
 * `MPCDAnalysis.LogicalEntityMeanSquareDisplacement`.
 */
class LogicalEntityMeanSquareDisplacement
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `mpcFluid_ == nullptr`.
	 * @throw OpenMPCD::InvalidConfigurationException
	 *        Throws if
	 *        `!isValidConfiguration(
	 *        	configuration.getSetting(
	 *        		"instrumentation.logicalEntityMeanSquareDisplacement"))`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `!mpcFluid_->numberOfParticlesPerLogicalEntityIsConstant()`.
	 *
	 * @param[in] configuration
	 *            The simulation configuration.
	 * @param[in] mpcFluid_
	 *            The fluid to measure. Must not be `nullptr`, and it must
	 *            return `true` when its
	 *            `numberOfParticlesPerLogicalEntityIsConstant` member is
	 *            called.
	 */
	LogicalEntityMeanSquareDisplacement(
		const OpenMPCD::Configuration& configuration,
		const OpenMPCD::CUDA::MPCFluid::Base* const mpcFluid_);

private:
	LogicalEntityMeanSquareDisplacement(
		const LogicalEntityMeanSquareDisplacement&);
			///< The copy constructor.

public:
	/**
	 * The destructor.
	 */
	~LogicalEntityMeanSquareDisplacement();

public:
	/**
	 * Returns whether the an attempt has been made to configure this class,
	 * i.e. whether the `instrumentation.logicalEntityMeanSquareDisplacement`
	 * configuration group exists.
	 *
	 * @param[in] config
	 *            The configuration to query.
	 */
	static bool isConfigured(const Configuration& config);

	/**
	 * Returns whether the given configuration group is a valid configuration.
	 *
	 * For a configuration group to be valid, it must
	 *  - contain the key `measureEveryNthSweep`, which must be a positive
	 *    integer
	 *  - contain the key `measurementArgumentCount`, which must be a
	 *    positive integer.
	 *
	 * @param[in] group
	 *            The configuration group to query.
	 */
	static bool isValidConfiguration(const Configuration::Setting& group);


public:
	/**
	 * Takes measurement data.
	 *
	 * This function is to be called by the `OpenMPCD::CUDA::Simulation` instance
	 * after every sweep.
	 */
	void measure();

	/**
	 * Returns, in units of `measurement time`, the maximum measurement time
	 * that is configured to be measured, i.e. \f$ N_A \f$.
	 */
	unsigned int getMaximumMeasurementTime() const;

	/**
	 * Returns `1` plus the maximum number `t` may take in
	 * `getMeanSquareDisplacement`.
	 */
	unsigned int getMeasurementCount() const;

	/**
	 * Returns the measured mean square displacement
	 * \f$ C \left( t, t + \Delta t \right) \f$ between measurement times
	 * \f$ t \f$ and \f$ T \f$.
	 *
	 * @note
	 * The times are given in units of the `measurement time`, which is
	 * described in the documentation of this class.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `t >= getMeasurementCount()`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `T >= getMeasurementCount()` or `T <= t` or
	 *        `T - t > getMaximumMeasurementTime()`.
	 *
	 * @param[in] t
	 *            The first measurement time \f$ t \f$. This value must be
	 *            smaller than `getMeasurementCount()`.
	 * @param[in] T
	 *            The second measurement time \f$ T \f$. This value must be
	 *            smaller than `getMeasurementCount()` and larger than `t`.
	 *            Also, `T - t` must not be larger than
	 *            `getMaximumMeasurementTime()`.
	 */
	MPCParticlePositionType getMeanSquareDisplacement(
		const unsigned int t,
		const unsigned int T) const;

	/**
	 * Saves the result to the given stream.
	 *
	 * For each value `t` in the range `[0, getMeasurementCount())`, and for
	 * each value `T` in the range `[t + 1, t + getMaximumMeasurementTime()]`
	 * (except for those where `T >= getMeasurementCount()`),
	 * a line will be written to the output stream, with the following fields,
	 * separated by tab characters:
	 * First, the current value of `t`, followed by the value of `T - t`,
	 * followed by the value of `getMeanSquareDisplacement(t, T)`.
	 *
	 * The numeric values will be written with precision
	 * <c>std::numeric_limits<OpenMPCD::FP>::digits10 + 2</c>.
	 *
	 * @param[out] stream
	 *             The stream to write to.
	 */
	void save(std::ostream& stream);

	/**
	 * Saves the result to the given run directory.
	 *
	 * The file within the given directory will be named
	 * `logicalEntityMeanSquareDisplacement.data`, and will be created if it
	 * does not exist, or truncated if it does. Parent directories will be
	 * created as needed.
	 * The file's contents will correspond to the output of
	 * `save(std::ostream&)`.
	 *
	 * @param[in] rundir
	 *            Path to the directory, into which the result file will be
	 *            written.
	 */
	void save(const std::string& rundir);

private:
	const LogicalEntityMeanSquareDisplacement& operator=(
		const LogicalEntityMeanSquareDisplacement&);
			///< The assignment operator.

private:
	const OpenMPCD::CUDA::MPCFluid::Base* const mpcFluid;
		///< The fluid measured.

	unsigned int measureEveryNthSweep;
		///< Specifies the snapshot interval.
	unsigned int measurementArgumentCount;
		///< Specifies the snapshot count.


	unsigned int sweepsSinceLastMeasurement;
		///< Counts the sweeps since the last measurement.

	std::vector<MPCParticlePositionType*> snapshots;
		///< The stored center of mass position snapshots.

	std::vector<std::vector<MPCParticlePositionType> >
		meanSquareDisplacements;
		/**< The measured mean square displacements. The first index specifies
		     the later measurement time, and the second index the difference
		     between later and prior measurement time.*/

	DeviceBuffer<MPCParticlePositionType> d_buffer;
		///< A buffer on the Device, used to store centers of mass.

}; //class LogicalEntityMeanSquareDisplacement
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_LOGICALENTITYMEANSQUAREDISPLACEMENT_HPP
