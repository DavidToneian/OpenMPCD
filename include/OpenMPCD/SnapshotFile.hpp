/**
 * @file
 * Defines the `OpenMPCD::SnapshotFile` class.
 */

#ifndef OPENMPCD_SNAPSHOTFILE_HPP
#define OPENMPCD_SNAPSHOTFILE_HPP

namespace OpenMPCD
{

/**
 * Base class for files that contain one or more simulation snapshots.
 */
class SnapshotFile
{
	protected:
		/**
		 * The constructor.
		 */
		SnapshotFile()
		{
		}

		/**
		 * The copy constructor.
		 */
		SnapshotFile(const SnapshotFile&);

	public:
		/**
		 * The destructor.
		 */
		virtual ~SnapshotFile()
		{
		}

	private:
		/**
		 * The assignment operator.
		 */
		const SnapshotFile& operator=(const SnapshotFile&);
}; //class SnapshotFile

} //namespace OpenMPCD

#endif /* OPENMPCD_SNAPSHOTFILE_HPP */
