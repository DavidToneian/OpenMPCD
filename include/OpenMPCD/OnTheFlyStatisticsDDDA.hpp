/**
 * @file
 * Defines the `OpenMPCD::OnTheFlyStatisticsDDDA` class.
 */

#ifndef OPENMPCD_ONTHEFLYSTATISTICSDDDA_HPP
#define OPENMPCD_ONTHEFLYSTATISTICSDDDA_HPP

#include <OpenMPCD/OnTheFlyStatistics.hpp>

#include <boost/optional/optional.hpp>

#include <vector>

namespace OpenMPCD
{

/**
 * Computes sample means and their errors for (possibly) serially correlated
 * data.
 *
 * The algorithm used is called "Dynamic Distributable Decorrelation
 * Algorithm" (DDDA), and is described in @cite Kent2007,
 * which in turn is partly based on @cite Flyvbjerg1989.
 *
 * @tparam T
 *         The data type. It must support addition and multiplication of two
 *         objects of this type, and division by an `unsigned long int`.
 */
template<typename T>
class OnTheFlyStatisticsDDDA
{
public:
	/**
	 * The constructor.
	 */
	OnTheFlyStatisticsDDDA();

public:
	/**
	 * Adds a datum to the sample.
	 *
	 * It is assumed that the "time" intervals between subsequently added data
	 * are constant; here, "time" may, for example, refer to Molecular Dynamics
	 * or Monte Carlo steps.
	 *
	 * @param[in] datum The datum to add.
	 */
	void addDatum(const T& datum);

	/**
	 * Returns the number of data points added so far.
	 */
	std::size_t getSampleSize() const;

	/**
	 * Returns the mean of all the values added so far.
	 *
	 * Since the mean of all values added is returned, and the sample size may
	 * not be a power of 2, statistics with different blocking length may
	 * not incorporate the same amount of information. This may lead to
	 * difficulties when using the error estimates of statistics of different
	 * block lengths to estimate the error in the entire, possibly correlated
	 * data set, since the statistics of different blocking lengths do not
	 * necessarily incorporate the same measurements.
	 *
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if no data have been added so far.
	 */
	const T getSampleMean() const;

	/**
	 * Returns the largest block size for which there is at least one data
	 * point.
	 *
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if no data have been added so far.
	 */
	std::size_t getMaximumBlockSize() const;

	/**
	 * Returns the ID of the largest block size created so far.
	 */
	std::size_t getMaximumBlockID() const;

	/**
	 * Returns whether the block with the given `blockID` has enough data to
	 * compute a sample variance.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `blockID` is out of
	 *        range.
	 *
	 * @param[in] blockID
	 *            The block ID, which must be in the range
	 *            `[0, getMaximumBlockID()]`.
	 */
	bool hasBlockVariance(const std::size_t blockID) const;

	/**
	 * Returns the sample variance in the block with the given `blockID`.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `blockID` is out of
	 *        range.
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if `!hasBlockVariance(blockID)`.
	 *
	 * @param[in] blockID
	 *            The block ID, which must be in the range
	 *            [0, `getMaximumBlockID()`].
	 */
	const T getBlockVariance(const std::size_t blockID) const;

	/**
	 * Returns the sample standard deviation in the block with the given
	 * `blockID`.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `blockID` is out of
	 *        range.
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if `!hasBlockVariance(blockID)`.
	 *
	 * @param[in] blockID
	 *            The block ID, which must be in the range
	 *            [0, `getMaximumBlockID()`].
	 */
	const T getBlockStandardDeviation(const std::size_t blockID) const;

	/**
	 * Returns the raw sample standard deviation, i.e. the sample standard
	 * deviation in block `0`.
	 *
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if `!hasBlockVariance(0)`.
	 */
	const T getSampleStandardDeviation() const;

	/**
	 * Returns an estimate for the standard deviation of the standard error of
	 * the mean for a given `blockID`.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `blockID` is out of
	 *        range.
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if `!hasBlockVariance(blockID)`.
	 *
	 * @param[in] blockID
	 *            The block ID, which must be in the range
	 *            [0, `getMaximumBlockID()`].
	 */
	const T getBlockStandardErrorOfTheMean(const std::size_t blockID) const;

	/**
	 * Returns an estimate for the standard deviation of the standard error of
	 * the mean for a given `blockID`.
	 *
	 * The returned estimate corresponds to Eq. (28) in @cite Flyvbjerg1989.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `blockID` is out of
	 *        range.
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if `!hasBlockVariance(blockID)`.
	 *
	 * @param[in] blockID
	 *            The block ID, which must be in the range
	 *            [0, `getMaximumBlockID()`].
	 */
	const T getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(
		const std::size_t blockID) const;

	/**
	 * Returns the block ID corresponding to the optimal block size, in the
	 * sense that the corresponding block provides the most accurate estimate
	 * for the standard error of the mean.
	 *
	 * If there is no variance in the data, `0` is returned.
	 *
	 * The algorithm used is described in Section IV of @cite Lee2011.
	 *
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if fewer than two data points have been added so far.
	 */
	std::size_t getOptimalBlockIDForStandardErrorOfTheMean() const;

	/**
	 * Returns whether the sample is large enough for the estimate of the
	 * standard error of the mean, as provided by the block indicated by
	 * `getOptimalBlockIDForStandardErrorOfTheMean`, to be reliable.
	 *
	 * The algorithm used is described in Section IV of @cite Lee2011.
	 *
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if fewer than two data points have been added so far.
	 */
	bool optimalStandardErrorOfTheMeanEstimateIsReliable() const;

	/**
	 * Returns the best estimation of the true standard error of the mean of
	 * the data, after decorrelation.
	 *
	 * @see optimalStandardErrorOfTheMeanEstimateIsReliable
	 *
	 * The algorithm used is described in Section IV of @cite Lee2011.
	 *
	 * @throw OpenMPCD::InvalidCallException
	 *        Throws if fewer than two data points have been added so far.
	 */
	const T getOptimalStandardErrorOfTheMean() const;

	/**
	 * Returns a string that contains the state of this instance.
	 *
	 * Since the serialized state is stored a string, it is representing the
	 * state only approximately.
	 *
	 * Currently, this is implemented only for the cases where
	 * <c> boost::is_arithmetic<T>::value </c> is `true`.
	 *
	 * @see unserializeFromString
	 */
	const std::string serializeToString() const;

	/**
	 * Discards the current state, and loads the state specified in the given
	 * string instead.
	 *
	 * Currently, this is implemented only for the cases where
	 * <c> boost::is_arithmetic<T>::value </c> is `true`.
	 *
	 * The user is responsible for attempting to load only such serialized
	 * states that have been generated with a compatible with `T`.
	 *
	 * @throw InvalidArgumentException
	 *        Throws if `state` does not encode a valid state.
	 *
	 * @param[in] state
	 *            The state to load. Must be a string created by
	 *            `serializeToString`.
	 */
	void unserializeFromString(const std::string& state);

private:
	/**
	 * Adds a datum to the sample to the given block, and propagates the datum
	 * to the next block level.
	 *
	 * @param[in] datum   The datum to add.
	 * @param[in] blockID The block to add the datum to.
	 */
	void addDatum(const T& datum, const std::size_t blockID);

private:
	std::vector<OnTheFlyStatistics<T> > blocks;
		///< Holds statistics for the time series blocks.
	std::vector<boost::optional<T> > waiting;
		///< Waiting data for higher-level blocks.
}; //class OnTheFlyStatisticsDDDA
} //namespace OpenMPCD

#include <OpenMPCD/ImplementationDetails/OnTheFlyStatisticsDDDA.hpp>

#endif //OPENMPCD_ONTHEFLYSTATISTICSDDDA_HPP
