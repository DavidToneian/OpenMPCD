/**
 * @file
 * Defines the MPCDAnalysis::File class.
 */

#ifndef MPCDANALYSIS_FILE_HPP
#define MPCDANALYSIS_FILE_HPP

#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <string>

namespace MPCDAnalysis
{
/**
 * Represents a possibly compressed file.
 */
class File
{
	public:
		/**
		 * The constructor.
		 *
		 * @param[in] filename The filename to try to open. If it does not exist, it is tried whether
		 *                     there is a filename with a compressed file's ending (e.g. ".bz2")
		 *                     there. If so, it is opened and returned.
		 * @throw OpenMPCD::IOException Throws if no compatible file was found.
		 */
		File(const std::string& filename);

		/**
		 * The destructor.
		 */
		~File();

	public:
		/**
		 * Returns an input stream for the file's decompressed data.
		 *
		 * This stream is valid only as long as this instance exists.
		 */
		std::istream& getInputStream();

		/**
		 * Returns the next line of decompressed data.
		 *
		 * @param[out] line The string to store the read line into.
		 * @return Returns whether a line has been read.
		 */
		bool readLine(std::string& line);

		/**
		 * Reads the name-value pairs of the file.
		 *
		 * Each line in the file that precedes the first line not starting with the '#' character
		 * is considered a comment. It is expected to have the form
		 * '# name = value'
		 * where name is a property's name, and value its corresponding value.
		 *
		 * @throw OpenMPCD::IOException Throws if the format is not adhered to.
		 */
		void readNameValuePairs();

		/**
		 * Returns whether there is a property of the given name.
		 *
		 * If readNameValuePairs has not been called before, this always returns false.
		 *
		 * @param[in] name The name of the property.
		 */
		bool hasProperty(const std::string& name) const;

		/**
		 * Returns the property of the given name.
		 *
		 * Call readNameValuePairs before to get results.
		 *
		 * @tparam ValueType The type of the value.
		 * @param[in] name The name of the property.
		 * @throw OpenMPCD::InvalidArgumentException Throws if there is no such property.
		 * @throw OpenMPCD::IOException              Throws if the property cannot be cast to the type specified.
		 */
		template<typename ValueType> const ValueType getProperty(const std::string& name) const;

	private:
		/**
		 * Initializes this instance with a regular file.
		 *
		 * @param[in] filename The filename to use.
		 */
		void initialize_regular(const std::string& filename);

		/**
		 * Initializes this instance with a bzip2-compressed file.
		 *
		 * @param[in] filename The filename to use.
		 */
		void initialize_bz2(const std::string& filename);

	private:
		std::deque<std::shared_ptr<std::istream>> streams; ///< The streams owned by this instance.

		std::map<std::string, std::string> nameValuePairs; ///< Holds a file's name-value pairs.

}; //class File
} //namespace MPCDAnalysis

#endif
