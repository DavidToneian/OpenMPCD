/**
 * @file
 * Defines the
 * `OpenMPCD::CUDA::MPCFluid::Instrumentation::NormalModeAutocorrelation`
 * class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_NORMALMODEAUTOCORRELATION_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_NORMALMODEAUTOCORRELATION_HPP


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
 * Measures the autocorrelation of normal coordinates in linear polymers.
 *
 * This class assumes that the MPC fluid it measures consists of linear polymer
 * chains. It can only be used if
 * `OpenMPCD::CUDA::MPCFluid::Base::numberOfParticlesPerLogicalEntityIsConstant`
 * is true.
 *
 * Let \f$ N_C \f$ correspond to the number of polymer chains in the fluid,
 * as returned by
 * `OpenMPCD::CUDA::MPCFluid::Base::getNumberOfLogicalEntities`,
 * and let the number of particles in each chain be \f$ N \f$, as returned by
 * `OpenMPCD::CUDA::MPCFluid::Base::getNumberOfParticlesPerLogicalEntity`.
 * Then, let \f$ \vec{q}_i^j \left( t \right) \f$ be the normal mode coordinate
 * of normal mode \f$ i \f$ of the chain \f$ j \f$ at simulation time \f$ t \f$,
 * as defined in `OpenMPCD::NormalMode`, with shift parameter \f$ S \f$.
 *
 * Then, this class samples, for all modes \f$ i \f$ and for configurable
 * correlation times \f$ \Delta t \f$, the quantity
 * \f[
 * 		C \left( t, t + \Delta t, i \right)
 * 		=
 * 		\frac{1}{N_C}
 * 		\sum_{j=1}^{N_C}
 * 		\vec{q}_i^j \left( t + \Delta t \right)
 * 		\cdot
 * 		\vec{q}_i^j \left( t \right)
 * \f]
 * via `OpenMPCD::CUDA::NormalMode::getAverageNormalCoordinateAutocorrelation`.
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
 * This class is configured via the `instrumentation.normalModeAutocorrelation`
 * configuration group. Within that group, `measureEveryNthSweep` defines
 * \f$ \tau \f$, and `autocorrelationArgumentCount` defines \f$ N_A \f$; given
 * those, the autocorrelation is measured at (simulation) correlation times
 * \f[
 * 		\Delta t \in
 * 		\left\{
 * 			0, \tau \Delta T, 2 \tau \Delta T,
 * 			\ldots, \left( N_A - 1 \right) \tau \Delta T
 * 		\right\}
 * \f]
 * where \f$ \Delta T \f$ is the simulation time between consecutive sweeps.
 * Consequently, in `measurement time`,
 * \f$ \Delta t \in \left\{ 0, 1, 2, \ldots, N_A - 1 \right\} \f$.
 *
 * Furthermore, the key `shift` within the
 * `instrumentation.normalModeAutocorrelation` configuration group specifies
 * the shift parameter \f$ S \f$ to be used (see `OpenMPCD::NormalMode` for the
 * definition of this parameter). If this key is not specified, it defaults to
 * `0.0`.
 *
 * For analysis of the produced results, see
 * `MPCDAnalysis.NormalModeAutocorrelation`.
 */
class NormalModeAutocorrelation
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
	 *        		"instrumentation.normalModeAutocorrelation"))`.
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
	NormalModeAutocorrelation(
		const OpenMPCD::Configuration& configuration,
		const OpenMPCD::CUDA::MPCFluid::Base* const mpcFluid_);

private:
	NormalModeAutocorrelation(const NormalModeAutocorrelation&);
		///< The copy constructor.

public:
	/**
	 * The destructor.
	 */
	~NormalModeAutocorrelation();

public:
	/**
	 * Returns whether the an attempt has been made to configure this class,
	 * i.e. whether the `instrumentation.normalModeAutocorrelation`
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
	 *  - contain the key `autocorrelationArgumentCount`, which must be a
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
	 * Returns, in units of `measurement time`, the maximum correlation time
	 * that is configured to be measured, i.e. \f$ N_A - 1 \f$.
	 */
	unsigned int getMaximumCorrelationTime() const;

	/**
	 * Returns `1` plus the maximum number `t` may take in `getAutocorrelation`.
	 */
	unsigned int getMeasurementCount() const;

	/**
	 * Returns the measured value of the autocorrelation at measurement times
	 * \f$ t \f$ and \f$ T \f$, and for normal mode index \f$ i \f$.
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
	 *        `T >= getMeasurementCount()` or `T < t` or
	 *        `T - t > getMaximumCorrelationTime()`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        Throws if `normalMode` is out of range.
	 *
	 * @param[in] t
	 *            The first measurement time \f$ t \f$. This value must be
	 *            smaller than `getMeasurementCount()`.
	 * @param[in] T
	 *            The second measurement time \f$ T \f$. This value must be
	 *            smaller than `getMeasurementCount()` and larger than or equal
	 *            to `t`. Also, `T - t` must not be larger than
	 *            `getMaximumCorrelationTime()`.
	 * @param[in] normalMode
	 *            The normal mode index \f$ i \f$, which must be less than or
	 *            equal to `mpcFluid->getNumberOfParticlesPerLogicalEntity()`,
	 *            where `mpcFluid` is the fluid passed to the constructor of
	 *            this instance.
	 */
	MPCParticlePositionType getAutocorrelation(
		const unsigned int t,
		const unsigned int T,
		const unsigned int normalMode) const;

	/**
	 * Saves the result to the given stream.
	 *
	 * For each value `t` in the range `[0, getMeasurementCount())`, and for
	 * each value `T` in the range `[t, t + getMaximumCorrelationTime()]`
	 * (except for those where `T >= getMeasurementCount()`),
	 * a line will be written to the output stream, with the following fields,
	 * separated by tab characters:
	 * First, the current value of `t`, followed by the value of `T - t`,
	 * followed by, for each normal mode index `n`, the value of
	 * `getAutocorrelation(t, T, n)`.
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
	 * `normalModeAutocorrelations.data`, and will be created if it does not
	 * exist, or truncated if it does. Parent directories will be created as
	 * needed.
	 * The file's contents will correspond to the output of
	 * `save(std::ostream&)`.
	 *
	 * @param[in] rundir
	 *            Path to the directory, into which the result file will be
	 *            written.
	 */
	void save(const std::string& rundir);

private:
	const NormalModeAutocorrelation& operator=(
		const NormalModeAutocorrelation&);
		///< The assignment operator.

private:
	const OpenMPCD::CUDA::MPCFluid::Base* const mpcFluid;
		///< The fluid measured.

	unsigned int measureEveryNthSweep;
		///< Specifies the snapshot interval.
	unsigned int autocorrelationArgumentCount;
		///< Specifies the snapshot count.


	unsigned int sweepsSinceLastMeasurement;
		///< Counts the sweeps since the last measurement.

	std::vector<DeviceBuffer<MPCParticlePositionType>*> snapshots;
		///< The stored normal mode coordinate snapshots.

	std::vector<std::vector<std::vector<MPCParticlePositionType> > >
		autocorrelations;
		/**< The measured autocorrelations. The first index specifies the later
		     measurement time, the second index the difference between later
		     and prior measurement time, and the third the normal mode index.*/

	MPCParticlePositionType shift;
		///< The normal mode shift parameter \f$ S \f$.

}; //class NormalModeAutocorrelation
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_NORMALMODEAUTOCORRELATION_HPP
