/**
 * @file
 * Implements functionality of the `OpenMPCD::Configuration` class.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_CONFIGURATION_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_CONFIGURATION_HPP

#include <OpenMPCD/Configuration.hpp>


namespace OpenMPCD
{

inline OpenMPCD::Configuration::Configuration(const std::string& filename)
	: config()
{
	try
	{
		config.readFile(filename.c_str());
	}
	catch(const std::exception& e)
	{
		if(std::string(e.what()) == "FileIOException")
		{
			std::string message = "Failed to open configuration file: ";
			message += filename;

			OPENMPCD_THROW(OpenMPCD::IOException, message);
		}

		if(std::string(e.what()) == "ParseException")
		{
			std::string message = "Failed to parse configuration file: ";
			message += filename;

			OPENMPCD_THROW(OpenMPCD::MalformedFileException, message);
		}

		throw;
	}
}

} //namespace OpenMPCD

#endif //OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_CONFIGURATION_HPP
