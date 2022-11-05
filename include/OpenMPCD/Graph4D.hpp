/**
 * @file
 * Defines the Graph4D class.
 */

#ifndef OPENMPCD_GRAPH4D_HPP
#define OPENMPCD_GRAPH4D_HPP

#include <OpenMPCD/Types.hpp>

#include <boost/tuple/tuple.hpp>
#include <deque>
#include <string>

namespace OpenMPCD
{
	/**
	 * Represents a 4D graph.
	 */
	class Graph4D
	{
		public:
			typedef std::deque<boost::tuple<double, double, double, double> > Container;
				///< The type the data points are collected in.

		public:
			/**
			 * Adds a data point.
			 * @param[in] w The w coordinate.
			 * @param[in] x The x coordinate.
			 * @param[in] y The y coordinate.
			 * @param[in] z The z coordinate.
			 */
			void addPoint(const FP w, const FP x, const FP y, const FP z)
			{
				points.push_back(boost::make_tuple(w, x, y, z));
			}

			/**
			 * Returns the data points.
			 */
			const Container& getPoints() const
			{
				return points;
			}

			/**
			 * Saves the graph data to the given file path.
			 * @param[in] path The file path to save to.
			 */
			void save(const std::string& path) const;

		private:
			Container points; ///< The data points.
	};
}

#endif
