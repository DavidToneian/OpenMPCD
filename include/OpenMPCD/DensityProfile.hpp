/**
 * @file
 * Defines the OpenMPCD::DensityProfile class.
 */

#ifndef OPENMPCD_DENSITYPROFILE_HPP
#define OPENMPCD_DENSITYPROFILE_HPP

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <string>
#include <vector>

namespace OpenMPCD
{
	/**
	 * Represents a density profile.
	 *
	 * This class can be configured via the `instrumentation.densityProfile`
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
	 */
	class DensityProfile
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
			DensityProfile(
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
			 * Increments the fill count.
			 */
			void incrementFillCount()
			{
				++fillCount;
			}

			/**
			 * Adds mass to the given point.
			 * The coordinates given to this function are the index of the cell and its subdivisions.
			 * @throw OutOfBoundsException If OPENMPCD_DEBUG is defined, throws if x, y, or z are too large.
			 * @param[in] x The x coordinate of the point.
			 * @param[in] y The y coordinate of the point.
			 * @param[in] z The z coordinate of the point.
			 * @param[in] m The mass to add.
			 */
			void add(const std::vector<std::vector<std::vector<FP> > >::size_type x,
			         const std::vector<std::vector<FP> >::size_type y,
			         const std::vector<FP>::size_type z,
			         const FP m)
			{
				#ifdef OPENMPCD_DEBUG
					if(x>=points.size())
						OPENMPCD_THROW(OutOfBoundsException, "x");
					if(y>=points[x].size())
						OPENMPCD_THROW(OutOfBoundsException, "y");
					if(z>=points[x][y].size())
						OPENMPCD_THROW(OutOfBoundsException, "z");
				#endif

				points[x][y][z] += m;
			}

			/**
			 * Saves the data into a file with the given path.
			 * Each line represents one point in the density profile.
			 * The first three columns are the x, y, and z coordinates of the point, respectively.
			 * The last column shows the averaged mass density at that point.
			 * @param[in] path The path to save to.
			 * @throw IOException Throws on an IO error.
			 */
			void saveToFile(const std::string& path) const;

		private:
			std::vector<std::vector<std::vector<FP> > > points; ///< The points in the flow profile.
			unsigned int fillCount; ///< The number of measurements performed.

			unsigned int cellSubdivisionsX; ///< The number of times collision cells are subdivided along the x direction.
			unsigned int cellSubdivisionsY; ///< The number of times collision cells are subdivided along the y direction.
			unsigned int cellSubdivisionsZ; ///< The number of times collision cells are subdivided along the z direction.
	};
}

#endif
