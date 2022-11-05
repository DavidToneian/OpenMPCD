/**
 * @file
 * Defines the Graph class.
 */

#ifndef OPENMPCD_GRAPH_HPP
#define OPENMPCD_GRAPH_HPP

#include <OpenMPCD/Types.hpp>

#include <deque>
#include <string>

namespace OpenMPCD
{
	/**
	 * Represents a 2D graph.
	 */
	class Graph
	{
		public:
			typedef std::deque<std::pair<FP, FP> > Container; ///< The type the data points are collected in.

		public:
			/**
			 * Adds a data point.
			 * @param[in] x The x coordinate.
			 * @param[in] y The y coordinate.
			 */
			void addPoint(const FP x, const FP y)
			{
				points.push_back(std::make_pair(x, y));
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
			 * @param[in] path                   The file path to save to.
			 * @param[in] prependGnuplotCommands Set to true to prepend gnuplot commands,
			 *                                   so that the resulting file can be plotted by calling gnuplot on it.
			 */
			void save(const std::string& path, const bool prependGnuplotCommands) const;

		private:
			Container points; ///< The data points.
	};
}

#endif
