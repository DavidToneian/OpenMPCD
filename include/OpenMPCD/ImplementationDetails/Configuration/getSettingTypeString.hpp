/**
 * @file
 * Defines the
 * `OpenMPCD::ImplementationDetails::Configuration::getSettingTypeString`
 * function.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_GETSETTINGTYPESTRING_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_GETSETTINGTYPESTRING_HPP

#include <OpenMPCD/Exceptions.hpp>

#include <libconfig.h++>
#include <string>

namespace OpenMPCD
{
namespace ImplementationDetails
{
namespace Configuration
{

/**
 * Returns a string corresponding to the name of the given type.
 * @param[in] type The type whose name to return.
 * @throw Exception Throws if the type is unknown.
 */
inline const std::string getSettingTypeString(const libconfig::Setting::Type type)
{
	switch(type)
	{
		case libconfig::Setting::TypeNone:
			return "TypeNone";

		case libconfig::Setting::TypeInt:
			return "TypeInt";

		case libconfig::Setting::TypeInt64:
			return "TypeInt64";

		case libconfig::Setting::TypeFloat:
			return "TypeFloat";

		case libconfig::Setting::TypeString:
			return "TypeString";

		case libconfig::Setting::TypeBoolean:
			return "TypeBoolean";

		case libconfig::Setting::TypeGroup:
			return "TypeGroup";

		case libconfig::Setting::TypeArray:
			return "TypeArray";

		case libconfig::Setting::TypeList:
			return "TypeList";

		default:
			OPENMPCD_THROW(Exception, "Unknown setting type.");
	}
}

} //namespace Configuration
} //namespace ImplementationDetails
} //namespace OpenMPCD

#endif
