/**
 * @file
 * Defines the `OpenMPCD::OnTheFlyStatistics` class.
 */

#ifndef OPENMPCD_ONTHEFLYSTATISTICS_HPP
#define OPENMPCD_ONTHEFLYSTATISTICS_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <cmath>
#include <string>

namespace OpenMPCD
{

/**
 * Computes sample means and variances "on-the-fly" or "online", i.e. without
 * storing the individual data points.
 *
 * For a description of the algorithm used, see "Formulas for Robust, One-Pass
 * Parallel Computation of Covariances and Arbitrary-Order Statistical Moments"
 * by Philippe Pébay, Sandia Report SAND2008-6212, 2008.
 *
 * @tparam T
 *         The data type. It must support addition and multiplication of two
 *         objects of this type, and division by an `unsigned long int`.
 */
template<typename T>
class OnTheFlyStatistics
{
	public:
		/**
		 * The constructor.
		 */
		OnTheFlyStatistics()
			: sampleSize(0), mean(0), varianceHelper(0)
		{
		}

	public:
		/**
		 * Returns the number of data points added so far.
		 */
		std::size_t getSampleSize() const
		{
			return sampleSize;
		}

		/**
		 * Returns the mean of all the values added so far.
		 *
		 * If no values have been added yet, returns `0`.
		 */
		const T getSampleMean() const
		{
			return mean;
		}

		/**
		 * Returns the unbiased sample variance of all the values added so far.
		 *
		 * The returned value contains Bessel's correction, i.e. the sum of
		 * squares of differences is divided by \f$ n - 1 \f$ rather than
		 * \f$ n \f$, where \$ n \f$ is the sample size.
		 *
		 * If fewer than two values have been added so far, returns `0`.
		 */
		const T getSampleVariance() const
		{
			if(getSampleSize() < 2)
				return 0;

			return varianceHelper / (getSampleSize() - 1);
		}

		/**
		 * Returns the unbiased sample standard deviation of all the values
		 * added so far.
		 *
		 * The returned value contains Bessel's correction, i.e. the sum of
		 * squares of differences is divided by \f$ n - 1 \f$ rather than
		 * \f$ n \f$, where \$ n \f$ is the sample size.
		 *
		 * If fewer than two values have been added so far, returns `0`.
		 */
		const T getSampleStandardDeviation() const
		{
			return sqrt(getSampleVariance());
		}

		/**
		 * Returns the standard error of the mean, i.e. the unbiased sample
		 * standard deviation divided by the square root of the sample size.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        If `OPENMPCD_DEBUG` is defined, throws if `getSampleSize()==0`.
		 */
		const T getStandardErrorOfTheMean() const
		{
			OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
				getSampleSize() != 0, InvalidCallException);

			return getSampleStandardDeviation() / sqrt(getSampleSize());
		}

		/**
		 * Adds a datum to the sample.
		 *
		 * @param[in] datum The datum to add.
		 */
		void addDatum(const T& datum)
		{
			//Since the division operator is used in in this function, allowing
			//integral data types here would lead to incorrect results. If
			//needed, one could provide a template specialization for this.
			BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);

			++sampleSize;

			const T delta(datum - mean);

			mean += delta / sampleSize;
			varianceHelper += delta * (datum - mean);
		}

		/**
		 * Returns a string that contains the state of this instance.
		 *
		 * Since the serialized state is stored a string, it is representing
		 * the state only approximately.
		 *
		 * Currently, this is implemented only for the cases where
		 * <c> boost::is_arithmetic<T>::value </c> is `true`.
		 *
		 * @see unserializeFromString
		 */
		const std::string serializeToString() const;

		/**
		 * Discards the current state, and loads the state specified in the
		 * given string instead.
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
		std::size_t sampleSize; ///< The number of data points sampled.
		T mean; ///< Holds the current arithmetic mean.
		T varianceHelper; ///< Helper value for the calculation of the variance.
}; //class OnTheFlyStatistics
}

#include <OpenMPCD/ImplementationDetails/OnTheFlyStatistics.hpp>

#endif //OPENMPCD_ONTHEFLYSTATISTICS_HPP
