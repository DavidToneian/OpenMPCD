/**
 * @file
 * Defines the `OpenMPCD::VTFSnapshotFile` class.
 */

#ifndef OPENMPCD_VTFSNAPSHOTFILE_HPP
#define OPENMPCD_VTFSNAPSHOTFILE_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/SnapshotFile.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <boost/optional.hpp>

#include <fstream>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace OpenMPCD
{

/**
 * Representation of a simulation snapshot file in the `VTF` format.
 *
 * The `VTF` format is documented at the following URL:
 * https://github.com/olenz/vtfplugin/wiki/VTF-format
 *
 * A snapshot file can be opened either in write mode or in read mode.
 * The instance is in read mode if it is passed a path to an existing file,
 * and in write mode otherwise.
 * In read mode, the snapshot cannot be changed (trying to do that results
 * in an instance of `OpenMPCD::InvalidCallException` being thrown).
 * In write mode, one cannot read data that does not belong to the structure
 * block.
 *
 * A VTF file starts with up to one structure block, followed by an arbitrary
 * number of timestep blocks.
 * After data that does not belong to the structure block has been supplied,
 * structure block information cannot be changed anymore; any attempts to do
 * so result in an instance of `OpenMPCD::InvalidCallException` being thrown.
 *
 * It is not guaranteed that data will be written to the snapshot file
 * immediately. Writes may be cached until the object is destroyed.
 */
class VTFSnapshotFile : public SnapshotFile
{
	public:
		/**
		 * Collection of properties of atoms.
		 */
		struct AtomProperties
		{
			boost::optional<FP> radius;        ///< The radius of the atom.
			boost::optional<std::string> name; ///< The name of the atom.
			boost::optional<std::string> type; ///< The type of the atom.
		};

	public:
		/**
		 * The constructor.
		 *
		 * The `path_` given will be used to open the snapshot file.
		 * If the file does not exist, it will be created, and the instance is
		 * in write mode. Otherwise, the instance is in read mode.
		 *
		 * @throw OpenMPCD::IOException
		 *        Throws if the file could not be opened or created.
		 * @throw OpenMPCD::MalformedFileException
		 *        In read mode, throws if the VTF file is malformed.
		 *
		 * @param[in] path_ The path to the snapshot file.
		 */
		VTFSnapshotFile(const std::string& path_);

		/**
		 * The destructor.
		 */
		virtual ~VTFSnapshotFile();

	public:
		/**
		 * Returns whether the instance is in write mode.
		 */
		bool isInWriteMode() const
		{
			return writeModeFlag;
		}

		/**
		 * Returns whether the instance is in read mode.
		 */
		bool isInReadMode() const
		{
			return !isInWriteMode();
		}

		/**
		 * Returns whether the structure block has been processed already.
		 *
		 * The structure block is read immediately when opening a file in read
		 * mode, and written when calling `writeTimestepBlock` for the first
		 * time, or destroying an instance in write mode.
		 */
		bool structureBlockHasBeenProcessed() const
		{
			return structureBlockProcessed;
		}

		/**
		 * Sets the size of the primary simulation volume.
		 *
		 * This information is part of the structure block.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been written.
		 *
		 * @param[in] x The size along the Cartesian `x` axis.
		 * @param[in] y The size along the Cartesian `y` axis.
		 * @param[in] z The size along the Cartesian `z` axis.
		 */
		void setPrimarySimulationVolumeSize(
			const FP& x, const FP& y, const FP& z);

		/**
		 * Returns whether the size of the primary simulation volume is set.
		 */
		bool primarySimulationVolumeSizeIsSet() const
		{
			return primarySimulationVolumeSizeSet;
		}

		/**
		 * Returns the size of the primary simulation volume.
		 *
		 * The returned vector holds the Cartesian coordinates.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if `!primarySimulationVolumeSizeIsSet()`.
		 */
		const Vector3D<FP>& getPrimarySimulationVolumeSize() const
		{
			if(!primarySimulationVolumeSizeIsSet())
				OPENMPCD_THROW(InvalidCallException, "");

			return primarySimulationVolumeSize;
		}

		/**
		 * Declares a number of atoms.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been written.
		 *
		 * @param[in] count The number of atoms to declare;
		 *                  if it is 0, nothing happens.
		 *
		 * @return Returns the ID of the first declared atom as the first value,
		 *         and the ID of the last declared atom as the second value.
		 *         If `count == 0`, the pair `{0, 0}` is returned.
		 */
		const std::pair<std::size_t, std::size_t>
		declareAtoms(const std::size_t count);

		/**
		 * Declares a number of atoms of the same kind.
		 *
		 * @throw OpenMPCD::InvalidArgumentException
		 *        Throws if the arguments violate the conditions outlined.
		 *
		 * @param[in] count  The number of atoms to declare;
		 *                   if it is 0, nothing happens.
		 * @param[in] radius The radius of the atoms, which must be
		 *                   greater than or equal to 0, or -1 for default.
		 * @param[in] name   The name of the atoms, as a string without
		 *                   whitespace and of length greater than 0 but
		 *                   less than or equal to 16, or empty for default.
		 * @param[in] type   The type of the atoms, as a string without
		 *                   whitespace and of length greater than 0 but
		 *                   less than or equal to 16, or empty for default.
		 *
		 * @return Returns the ID of the first declared atom as the first value,
		 *         and the ID of the last declared atom as the second value.
		 *         If `count == 0`, the pair `{0, 0}` is returned.
		 */
		const std::pair<std::size_t, std::size_t> declareAtoms(
			const std::size_t count, const FP radius,
			const std::string& name, const std::string& type);

		/**
		 * Returns the number of atoms that have been declared.
		 */
		std::size_t getNumberOfAtoms() const
		{
			if(atomRanges.empty())
				return 0;

			return atomRanges.back().last + 1;
		}

		/**
		 * Returns whether the given number is a valid atom ID, i.e.
		 * whether `atomID < getNumberOfAtoms()`.
		 *
		 * @param[in] atomID The atom ID to check.
		 */
		bool isValidAtomID(const std::size_t atomID) const
		{
			return atomID < getNumberOfAtoms();
		}

		/**
		 * Returns the properties of the given `atomID`.
		 *
		 * @throw OpenMPCD::OutOfBoundsException
		 *        Throws if `!isValidAtomID(atomID)`.
		 *
		 * @param[in] atomID The ID of the atom in question.
		 */
		const AtomProperties& getAtomProperties(const std::size_t atomID) const;

		/**
		 * Declares a bond between the two given atoms.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been written.
		 * @throw OpenMPCD::OutOfBoundsException
		 *        Throws if `!isValidAtomID(atom1)` or `!isValidAtomID(atom2)`.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        Throws if `atom1 == atom2`.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        Throws if a bond between those atoms exists already.
		 *
		 * @param[in] atom1 The ID of the first atom.
		 * @param[in] atom2 The ID of the second atom.
		 */
		void declareBond(std::size_t atom1, std::size_t atom2);

		/**
		 * Returns the set of bonds between atoms.
		 *
		 * Each entry `bond` of the returned value denotes a bond between the
		 * atoms with the IDs `bond.first` and `bond.second`, where
		 * `bond.first < bond.second`.
		 */
		const std::set<std::pair<std::size_t, std::size_t> >& getBonds() const
		{
			return bonds;
		}

		/**
		 * Returns whether the two given atoms share a bond.
		 *
		 * The order of the arguments does not influence the returned value.
		 *
		 * @throw OpenMPCD::OutOfBoundsException
		 *        Throws if `!isValidAtomID(atom1)` or `!isValidAtomID(atom2)`.
		 * @throw OpenMPCD::InvalidArgumentException
		 *        Throws if `atom1 == atom2`.
		 *
		 * @param[in] atom1 The ID of the first atom.
		 * @param[in] atom2 The ID of the second atom.
		 */
		bool hasBond(std::size_t atom1, std::size_t atom2) const;

		/**
		 * Starts a new timestep block, and writes the atom coordinates given.
		 *
		 * Calling this function will trigger writing of the structure block.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::NULLPointerException
		 *        Throws if `positions` is `nullptr`.
		 *
		 * @param[in] positions An array holding the positions of the
		 *                      `getNumberOfAtoms()` atoms, in the following
		 *                      format: First, there are the `x`, `y`, and `z`
		 *                      coordinates of atom `0`, in that order; then,
		 *                      the coordinates of atom `1` follow, etc.
		 * @param[in] velocities
		 *                      An array holding the velocities of the
		 *                      `getNumberOfAtoms()` atoms, in the following
		 *                      format: First, there are the `x`, `y`, and `z`
		 *                      velocities of atom `0`, in that order; then,
		 *                      the velocities of atom `1` follow, etc.
		 *                      If `nullptr` is passed, no velocities are
		 *                      written to the snapshot file.
		 */
		void writeTimestepBlock(
			const FP* const positions,
			const FP* const velocities = NULL);

		/**
		 * Reads the next timestep block.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 * @throw OpenMPCD::UnimplementedException
		 *        Throws if the timestep block is not in ordered format.
		 * @throw OpenMPCD::MalformedFileException
		 *        Throws if in the timestep block read, some atoms contain
		 *        velocity information, while others do not.
		 * @throw OpenMPCD::MalformedFileException
		 *        Throws if the timestep block is malformed or contains fewer
		 *        atoms than `getNumberOfAtoms()`.
		 *
		 * @param[out] positions An array capable of holding the positions of
		 *                       holding `3*getNumberOfAtoms()` elements, or
		 *                       `nullptr` if the positions are not desired. If
		 *                       not `nullptr`, the array will be populated in
		 *                       the following format: First, there are the `x`,
		 *                       `y`, and `z` coordinates of atom `0`, in that
		 *                       order; then, the coordinates of atom `1`
		 *                       follow, etc.
		 * @param[out] velocities
		 *                       An array capable of holding the velocities of
		 *                       holding `3*getNumberOfAtoms()` elements, or
		 *                       `nullptr` if the velocities are not desired. If
		 *                       not `nullptr`, the array will be populated in
		 *                       the following format: First, there are the `x`,
		 *                       `y`, and `z` velocities of atom `0`, in that
		 *                       order; then, the coordinates of atom `1`
		 *                       follow, etc.
		 *                       If no velocities are stored in the snapshot
		 *                       file, the given buffer's contents are left
		 *                       unchanged.
		 * @param[out] velocitiesEncountered
		 *                       If not `nullptr`, stores whether the timestep
		 *                       block read contained velocity information.
		 *
		 * @return Returns true if a complete timestep block has been read.
		 */
		bool readTimestepBlock(
			FP* const positions,
			FP* const velocities = NULL,
			bool* const velocitiesEncountered = NULL);


	private:
		/**
		 * Holds information about a range of atoms.
		 */
		struct AtomRange
		{
			std::size_t first;         ///< The first atom in this range.
			std::size_t last;          ///< The last atom in this range.
			AtomProperties properties; ///< The atom properties of this range.
		};


	private:
		/**
		 * Throws if the instance is not in write mode.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 */
		void assertWriteMode() const;

		/**
		 * Throws if the instance is not in read mode.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 */
		void assertReadMode() const;

		/**
		 * Throws if the structure block has already been processed.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been processed.
		 */
		void assertStructureBlockNotProcessed() const;

		/**
		 * Writes the structure block to the file.
		 *
		 * This also sets the `structureBlockProcessed` flag.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been written.
		 */
		void writeStructureBlock();

		/**
		 * Reads the snapshot file's structure block.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 * @throw OpenMPCD::MalformedFileException
		 *        Throws if the VTF file is malformed.
		 */
		void readStructureBlock();

		/**
		 * Reads a `unitcell line`.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 *
		 * @param[in] line The line to read.
		 */
		void readUnitcellLine(const std::string& line);

		/**
		 * Writes out the `atom lines`.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been written.
		 */
		void writeAtomLines();

		/**
		 * Reads an `atom line`.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 * @throw OpenMPCD::MalformedFileException
		 *        Throws if the given `line_` is malformed.
		 *
		 * @param[in] line_ The line to read.
		 */
		void readAtomLine(const std::string& line_);

		/**
		 * Saves the given atom range.
		 * @param[in] parameters The atom range to save.
		 */
		void setAtomRange(const AtomRange& range);

		/**
		 * Throws if the atom ranges are not contiguous and starting at 0.
		 *
		 * @throw OpenMPCD::Exception
		 *        Throws if the atom ranges are not contiguouos
		 *        and starting at 0.
		 */
		void assertAtomRangesContiguous() const;

		/**
		 * Writes out the `bond lines`.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in write mode.
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the structure block has already been written.
		 */
		void writeBondLines();

		/**
		 * Reads a `bond line`.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 * @throw OpenMPCD::MalformedFileException
		 *        Throws if the given `line_` is malformed.
		 *
		 * @param[in] line_ The line to read.
		 */
		void readBondLine(const std::string& line_);

		/**
		 * Returns the next line in the file.
		 *
		 * Leading whitespace will be removed.
		 *
		 * @throw OpenMPCD::InvalidCallException
		 *        Throws if the instance is not in read mode.
		 *
		 * @param[in] extract Whether to extract the line from the file stream.
		 */
		const std::string getLine(const bool extract);

		/**
		 * Returns the given string without leading whitespace.
		 *
		 * @param[in] str The string to strip from leading whitespace.
		 */
		static const std::string stripLeadingWhitespace(const std::string& str);

		/**
		 * Returns the numeric value contained in the given string.
		 *
		 * @throw OpenMPCD::MalformedFileException
		 *        Throws if the cast failed.
		 *
		 * @tparam T The type of the numeric value expected.
		 *
		 * @param[in] str The string to cast.
		 */
		template<typename T> static T lexicalCast(const std::string& str)
		{
			try
			{
				return boost::lexical_cast<T>(str);
			}
			catch(const boost::bad_lexical_cast&)
			{
				OPENMPCD_THROW(MalformedFileException, "Lexical cast failed.");
			}
		}


	private:
		std::fstream file; ///< The underlying file object.
		bool writeModeFlag; ///< Whether the instance is in write mode.
		bool structureBlockProcessed; /**< Whether the structure block has
		                                   already been processed. */

		bool primarySimulationVolumeSizeSet; /**< Whether information about
		                                          the size of the primary
		                                          simulation volume has been
		                                          supplied. */
		Vector3D<FP> primarySimulationVolumeSize; /**< The size of the primary
		                                               simulation volume. */

		AtomProperties defaultAtomProperties;
			///< Holds the current default vaues for new atoms.
		std::list<AtomRange> atomRanges; ///< The ranges of atoms defined.

		std::set<std::pair<std::size_t, std::size_t> > bonds;
			/**< The set of bonds. Each entry `bond` denotes a bond between
			     the atoms with the IDs `bond.first` and `bond.second`, where
			     `bond.first < bond.second`. */

}; //class VTFSnapshotFile

} //namespace OpenMPCD

#endif /* OPENMPCD_VTFSNAPSHOTFILE_HPP */
