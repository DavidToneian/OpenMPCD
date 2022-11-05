/**
 * @file
 * Defines utility functions.
 */

#ifndef MPCDANALYSIS_UTILITIES_HPP
#define MPCDANALYSIS_UTILITIES_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Configuration.hpp>

#include <boost/scoped_ptr.hpp>

#include <string>
#include <vector>

/**
 * Namespace for functionality to analyze OpenMPCD results.
 */
namespace MPCDAnalysis
{
/**
 * Contains utility functionality.
 */
namespace Utilities
{
	/**
	 * For the given valueKey, gets the configuration value for all the given rundirs if it is
	 * consistent, and throws otherwise.
	 *
	 * @tparam ValueType The expected value type.
	 * @param[in] rundirs  A list of run directories to consider.
	 * @param[in] valueKey The configuration key to the value to check and return.
	 * @throw OpenMPCD::InvalidArgumentException
	 *          Throws if rundirs is empty.
	 * @throw OpenMPCD::InvalidConfigurationException
	 *          Throws if the configuration value does not match ValueType.
	 * @throw OpenMPCD::InvalidConfigurationException
	 *          Throws if not all configurations agree on the value.
	 * @return Returns the common value.
	 */
	template<typename ValueType>
		const ValueType getConsistentConfigValue(
			const std::vector<std::string>& rundirs,
			const std::string& valueKey)
	{
		bool isFirst = true;
		ValueType value{};

		for(std::size_t i=0; i<rundirs.size(); ++i)
		{
			const std::string filename = rundirs[i] + "/config.txt";
			OpenMPCD::Configuration config(filename);

			const ValueType current = config.read<ValueType>(valueKey);

			if(isFirst)
			{
				isFirst = false;
				value = current;
			}
			else
			{
				if(value != current)
					OPENMPCD_THROW(OpenMPCD::InvalidConfigurationException,
						"Configuration value mismatch.");
			}
		}

		if(isFirst)
			OPENMPCD_THROW(OpenMPCD::InvalidArgumentException, "rundirs is empty.");

		return value;
	}

	/**
	 * Returns the names of files in the given directory that match the given filename pattern.
	 *
	 * This function does not search sub-directories.
	 * The returned filenames do not contain the directory path.
	 *
	 * @param[in] dirPath        The path to the directory to search in.
	 * @param[in] filenameRegex  The regular expression for filenames, not including the directory path.
	 * @param[in] prependDirPath Whether to prepend the directory path to the returned file names.
	 */
	const std::vector<std::string> getFilesByRegex(
		const std::string& dirPath, const std::string& filenameRegex, const bool prependDirPath = false);

} //namespace Utilities
} //namespace MPCDAnalysis

#endif
