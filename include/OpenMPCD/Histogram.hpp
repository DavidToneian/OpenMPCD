/**
 * @file
 * Defines the Histogram class.
 */

#ifndef OPENMPCD_HISTOGRAM_HPP
#define OPENMPCD_HISTOGRAM_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/Graph.hpp>
#include <OpenMPCD/Types.hpp>

#include <vector>

namespace OpenMPCD
{
	/**
	 * Represents a histogram.
	 */
	class Histogram
	{
		public:
			typedef unsigned long BinContent; ///< The type of bin contents.
			typedef std::vector<BinContent> Container; ///< The container for bins.

		public:
			/**
			 * The constructor.
			 * @param[in] name The histogram name.
			 * @param[in] conf The simulation configuration.
			 */
			Histogram(const std::string& name, const Configuration& conf);

			/**
			 * The constructor.
			 * @param[in] low      The lowest bin's lower end.
			 * @param[in] high     The highest bin's highest end.
			 * @param[in] binCount The number of bins.
			 */
			Histogram(const FP low, const FP high, const unsigned int binCount)
				: bins(binCount, 0), underflows(0), overflows(0), lowEnd(low), highEnd(high),
				  binSize((high-low)/binCount)
			{
			}

			/**
			 * Adds an entry to the histogram.
			 * @param[in] val The value to add.
			 */
			void fill(const FP val);

			/**
			 * Returns the number of underflows.
			 */
			BinContent getUnderflows() const
			{
				return underflows;
			}

			/**
			 * Returns the number of overflows.
			 */
			BinContent getOverflows() const
			{
				return overflows;
			}

			/**
			 * Returns the bins.
			 */
			const Container& getBins() const
			{
				return bins;
			}

			/**
			 * Returns the lower end of the histogram.
			 */
			FP getLowEnd() const
			{
				return lowEnd;
			}

			/**
			 * Returns the upper end of the histogram.
			 */
			FP getHighEnd() const
			{
				return highEnd;
			}

			/**
			 * Returns the bin size.
			 */
			FP getBinSize() const
			{
				return binSize;
			}

			/**
			 * Returns the integral of the histogram.
			 */
			FP getIntegral() const;

			/**
			 * Returns the graph corresponding to this histogram, with the area normalized to 1.
			 * @param[in] binPoint Since the bin has a non-zero width and the graph only consists of points,
			 *                     this parameter can be used to control where the graph point corresponding
			 *                     to a bin is situated. 0 means the point is at the leftmost end of the bin,
			 *                     1 corresponds to the rightmost end, and the values inbetween scale linearly.
			 * @throw InvalidArgumentException Throws if binPoint is not in the range [0,1].
			 */
			const Graph getNormalizedGraph(const FP binPoint=0.5) const;

			/**
			 * Saves the histogram at the given path.
			 * @param[in] filename The filename to save to.
			 * @param[in] binPoint Since the bin has a non-zero width and the graph only consists of points,
			 *                     this parameter can be used to control where the graph point corresponding
			 *                     to a bin is situated. 0 means the point is at the leftmost end of the bin,
			 *                     1 corresponds to the rightmost end, and the values inbetween scale linearly.
			 * @throw InvalidArgumentException Throws if binPoint is not in the range [0,1].
			 */
			void save(const std::string& filename, const FP binPoint=0.5) const;

		private:
			Container bins; ///< The histogram bins.
			BinContent underflows; ///< The number of underflows.
			BinContent overflows;  ///< The number of overflows.
			FP lowEnd;  ///< The lower end of the histogram.
			FP highEnd; ///< The higher end of the histogram.
			FP binSize; ///< The size of each bin.
	};
}

#endif
