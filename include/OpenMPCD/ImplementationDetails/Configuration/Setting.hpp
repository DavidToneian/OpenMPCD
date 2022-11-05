/**
 * @file
 * Defines the `OpenMPCD::ImplementationDetails::Configuration::Setting` class.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_SETTING_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_CONFIGURATION_SETTING_HPP

#include <OpenMPCD/Exceptions.hpp>

#include <libconfig.h++>

namespace OpenMPCD
{
/**
 * Namespace for implementation details that do not have to concern the users of the main functionality.
 */
namespace ImplementationDetails
{
/**
 * Namespace for implementation details of `OpenMPCD::Configuration`.
 */
namespace Configuration
{

/**
 * Helper class to handle libconfig++ settings.
 */
class Setting
{
	private:
		Setting(); ///< The constructor.

	public:
		/**
		 * Creates a new setting with the given value.
		 * @throw InvalidCallException Throws if there already is an exception of the given name.
		 * @param[in,out] parent The parent setting.
		 * @param[in]     name   The setting name.
		 * @param[in]     value  The value for the new setting.
		 */
		static void createSetting(libconfig::Setting* const parent, const std::string& name, const bool value)
		{
			throwIfNameInUse(*parent, name);

			libconfig::Setting& setting = parent->add(name, libconfig::Setting::TypeBoolean);
			setting = value;
		}

		/**
		 * Creates a new setting with the given value.
		 * @throw InvalidCallException Throws if there already is an exception of the given name.
		 * @param[in,out] parent The parent setting.
		 * @param[in]     name   The setting name.
		 * @param[in]     value  The value for the new setting.
		 */
		static void createSetting(libconfig::Setting* const parent, const std::string& name, const int value)
		{
			throwIfNameInUse(*parent, name);

			libconfig::Setting& setting = parent->add(name, libconfig::Setting::TypeInt);
			setting = value;
		}

		/**
		 * Creates a new setting with the given value.
		 * @throw InvalidCallException Throws if there already is an exception of the given name.
		 * @param[in,out] parent The parent setting.
		 * @param[in]     name   The setting name.
		 * @param[in]     value  The value for the new setting.
		 */
		static void createSetting(libconfig::Setting* const parent, const std::string& name, const long long value)
		{
			throwIfNameInUse(*parent, name);

			libconfig::Setting& setting = parent->add(name, libconfig::Setting::TypeInt64);
			setting = value;
		}

		/**
		 * Creates a new setting with the given value.
		 * @throw InvalidCallException Throws if there already is an exception of the given name.
		 * @param[in,out] parent The parent setting.
		 * @param[in]     name   The setting name.
		 * @param[in]     value  The value for the new setting.
		 */
		static void createSetting(libconfig::Setting* const parent, const std::string& name, const double value)
		{
			throwIfNameInUse(*parent, name);

			libconfig::Setting& setting = parent->add(name, libconfig::Setting::TypeFloat);
			setting = value;
		}

		/**
		 * Creates a new setting with the given value.
		 * @throw InvalidCallException Throws if there already is an exception of the given name.
		 * @param[in,out] parent The parent setting.
		 * @param[in]     name   The setting name.
		 * @param[in]     value  The value for the new setting.
		 */
		static void createSetting(libconfig::Setting* const parent, const std::string& name, const std::string& value)
		{
			throwIfNameInUse(*parent, name);

			libconfig::Setting& setting = parent->add(name, libconfig::Setting::TypeString);
			setting = value;
		}

		/**
		 * Creates a new setting with the given value.
		 * @throw InvalidCallException Throws if there already is an exception of the given name.
		 * @param[in,out] parent The parent setting.
		 * @param[in]     name   The setting name.
		 * @param[in]     value  The value for the new setting.
		 */
		static void createSetting(libconfig::Setting* const parent, const std::string& name, const char* const value)
		{
			throwIfNameInUse(*parent, name);

			libconfig::Setting& setting = parent->add(name, libconfig::Setting::TypeString);
			setting = value;
		}

	private:
		/**
		 * Throws `InvalidCallException` if the given parent setting has a child with the given name.
		 * @throw InvalidCallException Throws if the given parent setting has a child with the given name.
		 * @param[in] parent The parent to check.
		 * @param[in] name   The name in question.
		 */
		static void throwIfNameInUse(const libconfig::Setting& parent, const std::string& name)
		{
			if(parent.exists(name))
				OPENMPCD_THROW(InvalidCallException, "Setting name already in use.");
		}
}; //class Setting

} //namespace Configuration
} //namespace ImplementationDetails
} //namespace OpenMPCD

#endif
