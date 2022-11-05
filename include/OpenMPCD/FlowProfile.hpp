/**
 * @file
 * Defines the FlowProfile class.
 */

#ifndef OPENMPCD_FLOWPROFILE_HPP
#define OPENMPCD_FLOWPROFILE_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/OnTheFlyStatistics.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>

#include <string>

namespace OpenMPCD
{
	/**
	 * Represents a flow profile.
	 *
	 * This class can be configured via the `instrumentation.flowProfile`
	 * configuration group. If it is not present, this class will not be active.
	 * Otherwise, the following sub-settings are allowed:
	 * - `cellSubdivision` may be a group that has three positive integer
	 *    entries, `x`, `y`, and `z`, which define the resolution of the
	 *    profile. If, for example, `x` has a value of `3`, this means that each
	 *    collision cell is divided into three regions of equal length along the
	 *    \f$ x \f$ direction, and each region is sampled individually.
	 *    If `cellSubdivision` is not present, the values for
	 *    `cellSubdivision.x`, `cellSubdivision.y`, and `cellSubdivision.z`
	 *    default to `1`.
	 * - `sweepCountPerOutput` may be set to a non-negative integer. If the
	 *    value is greater than `0`, this instance measures the flow field for
	 *    `sweepCountPerOutput` successive sweeps, then computes statistical
	 *    data (averages, standard deviations) of the flow field as measured
	 *    during these sweeps, and finally starts a new measurement,
	 *    disregarding old data. If `sweepCountPerOutput` is `0` (the default),
	 *    all sweeps are measured, and the data taken contributes to the only
	 *    set of output data.
	 *
	 * @tparam T The data type.
	 */
	template<typename T> class FlowProfile
	{
		public:
			/**
			 * The constructor.
			 *
			 * @param[in] mpcBoxSizeX
			 *            The MPC simulation box size along the \f$ x \f$
			 *            direction.
			 * @param[in] mpcBoxSizeY
			 *            The MPC simulation box size along the \f$ y \f$
			 *            direction.
			 * @param[in] mpcBoxSizeZ
			 *            The MPC simulation box size along the \f$ z \f$
			 *            direction.
			 * @param[in] settings
			 *            The settings for this instance.
			 *
			 * @throw OpenMPCD::InvalidArgumentException
			 *        If `OPENMPCD_DEBUG` is defined, throws if any of the
			 *        numerical arguments is `0`.
			 *
			 * @throw OpenMPCD::InvalidConfigurationException
			 *        Throws if the configuration is invalid.
			 */
			FlowProfile(
				const unsigned int mpcBoxSizeX,
				const unsigned int mpcBoxSizeY,
				const unsigned int mpcBoxSizeZ,
				const Configuration::Setting& settings);

		public:
			/**
			 * Returns the number of times collision cells are subdivided along
			 * the \f$ x \f$ direction.
			 */
			unsigned int getCellSubdivisionsX() const
			{
				return cellSubdivisionsX;
			}

			/**
			 * Returns the number of times collision cells are subdivided along
			 * the \f$ y \f$ direction.
			 */
			unsigned int getCellSubdivisionsY() const
			{
				return cellSubdivisionsY;
			}

			/**
			 * Returns the number of times collision cells are subdivided along
			 * the \f$ z \f$ direction.
			 */
			unsigned int getCellSubdivisionsZ() const
			{
				return cellSubdivisionsZ;
			}

			/**
			 * Adds data to the given point.
			 *
			 * The coordinates given to this function are the index of the cell
			 * and its subdivisions.
			 *
			 *
			 * @throw OpenMPCD::InvalidCallException
			 *        If `OPENMPCD_DEBUG` is defined, throws if `newSweep` has
			 *        not been called before.
			 * @throw OpenMPCD::OutOfBoundsException
			 *        If `OPENMPCD_DEBUG` is defined, throws if `x`, `y`, or `z`
			 *        are too large, that is, if
			 *        `x >= mpcBoxSizeX * getCellSubdivisionsX()`, where
			 *        `mpcBoxSizeX` is the size of the simulation box along the
			 *        \f$ x \f$ direction, and analogously for the other
			 *        coordinates.
			 *
			 * @param[in] x The x coordinate of the point.
			 * @param[in] y The y coordinate of the point.
			 * @param[in] z The z coordinate of the point.
			 * @param[in] v The vector to add.
			 */
			void add(const std::size_t x, const std::size_t y,
			         const std::size_t z, const Vector3D<T>& v)
			{
				OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
					!outputBlocks.empty(), InvalidCallException);

				OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
					x < outputBlocks.back()->shape()[0], OutOfBoundsException);
				OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
					y < outputBlocks.back()->shape()[1], OutOfBoundsException);
				OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
					z < outputBlocks.back()->shape()[2], OutOfBoundsException);

				(*outputBlocks.back())[x][y][z][0].addDatum(v.getX());
				(*outputBlocks.back())[x][y][z][1].addDatum(v.getY());
				(*outputBlocks.back())[x][y][z][2].addDatum(v.getZ());
			}

			/**
			 * Signals that a new sweep has begun.
			 */
			void newSweep();

			/**
			 * Saves the data into a file with the given path.
			 *
			 * The first line is a comment header that starts with a `#`
			 * character and contains labels for each of the columns, which are
			 * described below.
			 *
			 * What follows after this first line is a number of output blocks,
			 * where the blocks are separated from one another by blank lines
			 * (but there is neither a blank line between the header and the
			 * first block, nor between the last block and the end of the file.)
			 *
			 * Each block represents a number of sweeps that have been measured
			 * in succession. What the block contains are statistical data
			 * about this series of sweeps, as described below. The number of
			 * sweeps per block is configured in the `sweepCountPerOutput`
			 * configuration option. The last block of an output file may
			 * contain fewer sweeps, if the simulation terminates before an
			 * integer multiple of `sweepCountPerOutput` could be reached.
			 *
			 * Each line in a block represents one point in the flow field.
			 * The first three columns are the x, y, and z coordinates of the
			 * point, respectively.
			 * The next three columns are the means of the x, y, and z
			 * coordinates of the velocity.
			 * The next three columns are the standard deviations of the x, y,
			 * and z coordinates of the velocity.
			 * The last column contains the sample sizes.
			 *
			 * @throw OpenMPCD::IOException Throws on an IO error.
			 *
			 * @param[in] path The path to save to.
			 */
			void saveToFile(const std::string& path) const;

		private:
			/**
			 * Creates a new output block.
			 */
			void createOutputBlock();

			/**
			 * Prints the given output block to the given stream.
			 *
			 * @param[in]  outputBlock
			 *             The output block to print.
			 * @param[out] stream
			 *             The stream to print to.
			 */
			void printOutputBlockToStream(
				const boost::multi_array<OnTheFlyStatistics<T>, 4 >& outputBlock,
				std::ostream& stream) const;

		private:
			const unsigned int simulationBoxSizeX;
				/**< The size of the simulation box along the \f$ x \f$
				     direction.*/
			const unsigned int simulationBoxSizeY;
				/**< The size of the simulation box along the \f$ y \f$
				     direction.*/
			const unsigned int simulationBoxSizeZ;
				/**< The size of the simulation box along the \f$ z \f$
				     direction.*/

			unsigned int cellSubdivisionsX;
				/**< The number of times collision cells are subdivided along
				     the \f$ x \f$ direction.*/
			unsigned int cellSubdivisionsY;
				/**< The number of times collision cells are subdivided along
				     the \f$ y \f$ direction.*/
			unsigned int cellSubdivisionsZ;
				/**< The number of times collision cells are subdivided along
				     the \f$ z \f$ direction.*/

			unsigned int sweepCountPerOutput;
				///< How many sweeps should enter into one output.
			unsigned int currentBlockSweepCount;
				///< The number of sweeps in the current output block.

			std::deque<
				boost::shared_ptr<boost::multi_array<OnTheFlyStatistics<T>, 4>
				> >
					outputBlocks; ///< Pointers to output blocks.
	};
}

#endif
