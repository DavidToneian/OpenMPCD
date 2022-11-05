/**
 * @file
 * Defines the Configuration class
 */

#ifndef OPENMPCD_CONFIGURATION_HPP
#define OPENMPCD_CONFIGURATION_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/ImplementationDetails/Configuration/copy.hpp>
#include <OpenMPCD/ImplementationDetails/Configuration/getSettingTypeString.hpp>
#include <OpenMPCD/ImplementationDetails/Configuration/Setting.hpp>
#include <OpenMPCD/Utility/CompilerDetection.hpp>

#include <boost/algorithm/string.hpp>
#include <libconfig.h++>

#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace OpenMPCD
{
	/**
	 * Represents the configuration of the simulation.
	 */
	class Configuration
	{
		public:
			class List;

			/**
			 * Represents a setting in the configuration.
			 */
			class Setting
			{
			public:
				/**
				 * The constructor.
				 * The instance is only valid as long as the Configuration
				 * instance it originated from is valid.
				 *
				 * @param[in] s The setting.
				 */
				Setting(const libconfig::Setting& s) : setting(&s)
				{
				}

				/**
				 * Returns whether the setting has a name.
				 */
				bool hasName() const
				{
					return setting->getName() != NULL;
				}

				/**
				 * Returns the name of the setting.
				 *
				 * @throw OpenMPCD::InvalidCallException
				 *        If `OPENMPCD_DEBUG` is defined, throws if `!hasName()`.
				 */
				const std::string getName() const
				{
					#ifdef OPENMPCD_DEBUG
						if(!hasName())
							OPENMPCD_THROW(InvalidCallException, "");
					#endif

					return setting->getName();
				}

				/**
				 * Returns whether a setting with the given name exists.
				 *
				 * @throw OpenMPCD::InvalidArgumentException
				 *        Throws if the setting name is illegal.
				 *
				 * @param[in] settingName The setting name.
				 */
				bool has(const std::string& settingName) const
				{
					const std::string::size_type dot = settingName.find(".");

					if(dot != std::string::npos)
					{
						if(dot == 0 || dot + 1 >= settingName.size())
						{
							OPENMPCD_THROW(
								InvalidArgumentException,
								settingName + " is not a legal setting name.");
						}

						const std::string first = settingName.substr(0, dot);

						if(!setting->exists(first))
							return false;

						const Setting subsetting(setting->lookup(first));

						const std::string second = settingName.substr(dot + 1);
						return subsetting.has(second);
					}

					return setting->exists(settingName);
				}

				/**
				 * Reads the specified setting and stores it at the given
				 * location.
				 *
				 * @throw OpenMPCD::InvalidArgumentException
				 *        Throws if the setting name is illegal.
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the setting does not exist.
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the setting could not be read into the
				 *        specified type.
				 * @throw OpenMPCD::NULLPointerException
				 *        If `OPENMPCD_DEBUG` is defined, throws if the given
				 *        pointer is `NULL`.
				 *
				 * @tparam ValueType The type of the settings value.
				 *
				 * @param[in]  name  The setting name, relative to this setting.
				 * @param[out] value The location to read the setting value
				 *                   into.
				 */
				template<typename ValueType>
				void read(const std::string& name, ValueType* const value)
				const
				{
					#ifdef OPENMPCD_DEBUG
						if(value==NULL)
							OPENMPCD_THROW(NULLPointerException, "`value`");
					#endif

					const std::string::size_type dot = name.find(".");

					if(dot != std::string::npos)
					{
						if(dot == 0 || dot + 1 >= name.size())
						{
							OPENMPCD_THROW(
								InvalidArgumentException,
								name + " is not a legal setting name.");
						}

						const std::string first = name.substr(0, dot);

						if(!setting->exists(first))
						{
							OPENMPCD_THROW(
								InvalidConfigurationException,
								name + " does not exist.");
						}

						const Setting subsetting(setting->lookup(first));

						const std::string second = name.substr(dot + 1);
						return subsetting.read(second, value);
					}

					if(!setting->lookupValue(name, *value))
					{
						if(!setting->exists(name))
							OPENMPCD_THROW(
								InvalidConfigurationException,
								name+" does not exist.");

						const libconfig::Setting::Type type =
							setting->lookup(name).getType();
						const std::string typeString =
							ImplementationDetails::Configuration::
							getSettingTypeString(type);

						OPENMPCD_THROW(
							InvalidConfigurationException,
							name + " has an unexpected type: " + typeString);
					}
				}

				/**
				 * Reads the specified setting and stores it at the given
				 * location.
				 *
				 * @throw OpenMPCD::InvalidArgumentException
				 *        Throws if the setting name is illegal.
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the setting does not exist.
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the setting could not be read into the
				 *        specified type.
				 * @throw OpenMPCD::NULLPointerException
				 *        If `OPENMPCD_DEBUG` is defined, throws if the given
				 *        pointer is `NULL`.
				 *
				 * @param[in]  name  The setting name, relative to this setting.
				 * @param[out] value The location to read the setting value
				 *                   into.
				 */
				void read(const std::string& name, std::size_t* const value)
				const
				{
					unsigned int tmp;
					read(name, &tmp);
					*value = tmp;
				}

				/**
				 * Returns the specified setting.
				 *
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the settings could not be read into the
				 *        specified type.
				 *
				 * @tparam ValueType The type of the settings value.
				 *
				 * @param[in] name The setting name, relative to this setting.
				 */
				template<typename ValueType>
				ValueType read(const std::string& name) const
				{
					ValueType value;
					read(name, &value);
					return value;
				}

				/**
				 * Returns the number of direct child settings in this setting.
				 *
				 * This function does not count children of children.
				 */
				std::size_t getChildCount() const
				{
					return static_cast<std::size_t>(setting->getLength());
				}

				/**
				 * Returns the child setting with the given index.
				 *
				 * @param[in] childIndex
				 *            The index of the child setting, which must be less
				 *            than `getChildCount()`.
				 *
				 * @throw OpenMPCD::InvalidArgumentException
				 *        Throws if the index is illegal.
				 */
				const Setting getChild(const std::size_t childIndex) const
				{
					typedef libconfig::Setting::const_iterator It;

					std::size_t current = 0;
					for(It it = setting->begin(); it != setting->end(); ++it)
					{
						if(current == childIndex)
							return Setting(*it);
						++current;
					}

					OPENMPCD_THROW(InvalidArgumentException, "`childIndex`");
				}

				/**
				 * Returns the setting object with the given name.
				 *
				 * @throw OpenMPCD::InvalidArgumentException
				 *        Throws if the setting name is illegal.
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the given name does not exist.
				 *
				 * @param[in] name The setting name.
				 */
				const Setting getSetting(const std::string& name) const
				{
					if(!has(name))
					{
						OPENMPCD_THROW(
							InvalidConfigurationException,
							name + " does not exist.");
					}

					return Setting(setting->lookup(name));
				}

				/**
				 * Returns the list object with the given name.
				 *
				 * @throw OpenMPCD::InvalidArgumentException
				 *        Throws if the setting name is illegal.
				 * @throw OpenMPCD::InvalidConfigurationException
				 *        Throws if the given name does not exist.
				 *
				 * @param[in] name The setting name.
				 */
				const List getList(const std::string& name) const;

				/**
				 * Returns whether all children in this setting have names that
				 * are in the given set.
				 *
				 * @param[in]  names
				 *             The names against to compare the children's names
				 *             to.
				 * @param[out] offender
				 *             If not `NULL` and if `true` is returned, the
				 *             pointee is set to the name of the first child
				 *             encountered whose name is not in the given set
				 *             `names`.
				 */
				bool childrenHaveNamesInCollection(
					const std::set<std::string>& names,
					std::string* const offender = NULL) const;

			private:
				const libconfig::Setting* setting; ///< The setting.
			}; //class Setting

			/**
			 * Represents a list, or an array, of values.
			 */
			class List
			{
				public:
					/**
					 * The constructor.
					 * The instance is only valid as long as the Configuration instance it originated
					 * from is valid.
					 * @param[in] s The setting.
					 * @throw Exception If OPENMPCD_DEBUG is defined, throws if this setting is
					 *                  neither a list nor an array in the libconfig sense.
					 */
					List(const libconfig::Setting& s) : setting(&s)
					{
						#ifdef OPENMPCD_DEBUG
							if(!setting->isArray() && !setting->isList())
								OPENMPCD_THROW(Exception, "Unexpected setting type.");
						#endif
					}

				public:
					/**
					 * Returns the number of elements in the list.
					 */
					unsigned int getSize() const
					{
						return setting->getLength();
					}

					/**
					 * Returns the setting object with the given index.
					 *
					 * @throw OpenMPCD::InvalidConfigurationException
					 *        Throws if the given index does not exist.
					 *
					 * @param[in] index The setting index, starting at 0.
					 */
					const Setting getSetting(const unsigned int index) const
					{
						if(index >= getSize())
						{
							OPENMPCD_THROW(
								InvalidConfigurationException, "index");
						}

						return Setting((*setting)[index]);
					}

					/**
					 * Returns the list with the given index.
					 * @param[in] index The index to return.
					 * @throw InvalidConfigurationException Throws if the given index does not exist
					 *                                      or is not a list.
					 */
					const List getList(const unsigned int index) const
					{
						if(index >= getSize())
							OPENMPCD_THROW(InvalidConfigurationException, "index");

						const libconfig::Setting& child = (*setting)[index];

						if(!child.isArray() && !child.isList())
							OPENMPCD_THROW(Exception, "Tried to getList, but found something else.");

						return List(child);
					}

					/**
					 * Reads the specified setting and stores them in the given location.
					 * @throw InvalidConfigurationException Throws if the setting could not be read into the
					 *                                      specified type.
					 * @throw NULLPointerException          If OPENMPCD_DEBUG is defined, throws if the given
					 *                                      pointer is NULL.
					 * @tparam ValueType The type of the settings value.
					 * @param[in]  index The setting index.
					 * @param[out] value The location to read the setting value into.
					 */
					template<typename ValueType>
						void read(const unsigned int index, ValueType* const value) const
					{
						#ifdef OPENMPCD_DEBUG
							if(value==NULL)
								OPENMPCD_THROW(NULLPointerException, "value");
						#endif

						try
						{
							*value=static_cast<ValueType>((*setting)[index]);
						}
						catch(...)
						{
							OPENMPCD_THROW(
								InvalidConfigurationException,
								"Setting has an unexpected type or does not exist.");
						}
					}

					/**
					 * Reads the specified setting and stores it in the given
					 * location.
					 *
					 * @throw OpenMPCD::InvalidConfigurationException
					 *        Throws if the setting ist not a string.
					 * @throw OpenMPCD::NULLPointerException
					 *        If `OPENMPCD_DEBUG` is defined, throws if
					 *        `value == nullptr`.
					 *
					 * @param[in]  index
					 *             The setting index.
					 * @param[out] value
					 *             The location to read the setting value into.
					 */
					void read(
						const unsigned int index, std::string* const value)
					const
					{
						#ifdef OPENMPCD_DEBUG
							if(value==NULL)
								OPENMPCD_THROW(NULLPointerException, "value");
						#endif

						try
						{
							*value = (*setting)[index].operator std::string();
						}
						catch(...)
						{
							OPENMPCD_THROW(
								InvalidConfigurationException,
								"Setting has an unexpected type or does not "
								"exist.");
						}
					}

					/**
					 * Returns the specified setting from the given configuration file.
					 * @throw InvalidConfigurationException Throws if the settings could not be read into
					 *                                      the specified type.
					 * @tparam ValueType The type of the settings value.
					 * @param[in] index The setting index
					 */
					template<typename ValueType>
						ValueType read(const unsigned int index) const
					{
						ValueType value;
						read(index, &value);
						return value;
					}

				private:
					const libconfig::Setting* setting; ///< The underlying setting.
			};

		public:
			/**
			 * The constructor.
			 */
			Configuration()
			{
			}

			/**
			 * The constructor.
			 *
			 * @throw OpenMPCD::IOException
			 *        Throws if the configuration file could not be read.
			 * @throw OpenMPCD::MalformedFileException
			 *        Throws if the configuration file is malformed.
			 *
			 * @param[in] filename The filename for the configuration file.
			 */
			Configuration(const std::string& filename);

			/**
			 * The copy constructor.
			 * @param[in] rhs The instance to copy.
			 */
			Configuration(const Configuration& rhs)
			{
				ImplementationDetails::Configuration::copy(rhs.config, &config);
			}

		public:
			/**
			 * Returns whether a setting with the given name exists.
			 * @param[in] setting The setting name.
			 */
			bool has(const std::string& setting) const
			{
				return config.exists(setting);
			}

			/**
			 * Reads the specified settings from the given configuration file and stores them in the given location.
			 * @throw std::runtime_error Throws if any of the settings could not be read into the specified type.
			 * @throw std::runtime_error If OPENMPCD_DEBUG is defined, throws if any of given pointers is NULL.
			 * @tparam ValueType     The types of the settings values.
			 * @tparam settingsCount The number of settings to read.
			 * @param[in,out] settingsAndValues The names of the settings to read, and the locations to store their values into.
			 */
			template<typename ValueType, unsigned int settingsCount>
				void read(const std::pair<const char*, ValueType*> (&settingsAndValues)[settingsCount]) const
			{
				for(unsigned int i=0; i<settingsCount; ++i)
					read(settingsAndValues[i].first, settingsAndValues[i].second);
			}

			/**
			 * Reads the specified setting from the given configuration file and stores them in the given location.
			 * @throw std::runtime_error Throws if the settings could not be read into the specified type.
			 * @throw std::runtime_error If OPENMPCD_DEBUG is defined, throws if the given pointer is NULL.
			 * @tparam ValueType The type of the settings value.
			 * @param[in]  setting The name of the setting to read.
			 * @param[out] value   The location to read the setting value into.
			 */
			template<typename ValueType>
				void read(const std::string& setting, ValueType* const value) const
			{
				#ifdef OPENMPCD_DEBUG
					if(value==NULL)
						OPENMPCD_THROW(NULLPointerException, "value");
				#endif

				if(!config.lookupValue(setting, *value))
				{
					if(!config.exists(setting))
						OPENMPCD_THROW(InvalidConfigurationException, setting+" does not exist.");

					const libconfig::Setting::Type type = config.lookup(setting).getType();
					const std::string typeString = ImplementationDetails::Configuration::getSettingTypeString(type);

					OPENMPCD_THROW(InvalidConfigurationException, setting+" has an unexpected type: " + typeString);
				}
			}

			/**
			 * Returns the specified setting from the given configuration file.
			 * @throw std::runtime_error Throws if the settings could not be read into the specified type.
			 * @tparam ValueType The type of the settings value.
			 * @param[in] setting The name of the setting to read.
			 */
			template<typename ValueType>
				ValueType read(const std::string& setting) const
			{
				ValueType value;
				read(setting, &value);
				return value;
			}

			/**
			 * Returns the setting object with the given name.
			 *
			 * @throw OpenMPCD::InvalidConfigurationException
			 *        Throws if the given name does not exist.
			 *
			 * @param[in] name The setting name.
			 */
			const Setting getSetting(const std::string& name) const
			{
				if(!has(name))
				{
					OPENMPCD_THROW(
						InvalidConfigurationException, name+" does not exist.");
				}

				return Setting(config.lookup(name));
			}

			/**
			 * Returns the list with the given name.
			 * @param[in] name The list name.
			 * @throw InvalidConfigurationException Throws if the given name does not exist
			 *                                      or is not a list.
			 */
			const List getList(const std::string& name) const
			{
				if(!has(name))
					OPENMPCD_THROW(InvalidConfigurationException, name+" does not exist.");

				const libconfig::Setting& child = config.lookup(name);

				if(!child.isArray() && !child.isList())
					OPENMPCD_THROW(Exception, "Tried to getList, but found something else.");

				return List(child);
			}

			/**
			 * Asserts that the given setting exists, and has the specified type and value.
			 * @throw std::runtime_error Throws if the assertion fails.
			 * @tparam ValueType The type of the settings value.
			 * @param[in] setting The name of the setting to read.
			 * @param[in] value   The value that the setting should have.
			 */
			template <typename ValueType>
				void assertValue(const std::string& setting, const ValueType& value) const
			{
				ValueType configValue;
				if(!config.lookupValue(setting, configValue))
					OPENMPCD_THROW(InvalidConfigurationException, setting);

				if(configValue!=value)
				{
					std::stringstream message;
					message<<setting<<": value is \""<<configValue<<"\", expected \""<<value<<"\"";
					OPENMPCD_THROW(InvalidConfigurationException, message.str());
				}
			}

			/**
			 * Asserts that the given setting exists, and has the specified type and value.
			 * @throw std::runtime_error Throws if the assertion fails.
			 * @tparam ValueType The type of the settings value.
			 * @param[in] setting The name of the setting to read.
			 * @param[in] value   The value that the setting should have.
			 */
			template <int N>
				void assertValue(const std::string& setting, const char (&value)[N]) const
			{
				std::string configValue;
				if(!config.lookupValue(setting, configValue))
					OPENMPCD_THROW(InvalidConfigurationException, setting);

				if(configValue!=value)
				{
					std::stringstream message;
					message<<setting<<": value is \""<<configValue<<"\", expected \""<<value<<"\"";
					OPENMPCD_THROW(InvalidConfigurationException, message.str());
				}
			}

			/**
			 * Sets a setting's value.
			 * @throw InvalidArgumentException Throws if the setting already exists, but is of the wrong type.
			 * @tparam ValueType The type of the setting's value.
			 * @param[in] setting The name of the setting to set.
			 * @param[in] value   The value to set the setting to.
			 */
			template<typename ValueType>
				void set(const std::string& setting, const ValueType value)
			{
				std::vector<std::string> pathComponents;
				boost::algorithm::split(pathComponents, setting, boost::algorithm::is_any_of("."));

				libconfig::Setting* currentSetting = &config.getRoot();
				for(unsigned int level=0; level < pathComponents.size(); ++level)
				{
					const bool isLastLevel = level + 1 == pathComponents.size();
					const std::string currentName = pathComponents[level];

					if(isLastLevel)
					{
						if(currentSetting->exists(currentName))
						{
							try
							{
								(*currentSetting)[currentName.c_str()] = value;
							}
							catch(const libconfig::SettingTypeException&)
							{
								OPENMPCD_THROW(InvalidArgumentException, "Tried to set existing setting with wrong type.");
							}
						}
						else
						{
							ImplementationDetails::Configuration::Setting::createSetting(currentSetting, currentName, value);
						}
					}
					else
					{
						if(currentSetting->exists(currentName))
						{
							std::string lookupPath = currentSetting->getPath();
							if(!lookupPath.empty())
								lookupPath += ".";
							lookupPath += currentName;

							currentSetting = &config.lookup(lookupPath);
							#ifdef OPENMPCD_DEBUG
								if(!currentSetting->isAggregate())
									OPENMPCD_THROW(Exception, lookupPath + " is not aggregate.");
							#endif
						}
						else
						{
							currentSetting = &currentSetting->add(currentName, libconfig::Setting::TypeGroup);
						}
					}
				}
			}

			/**
			 * Creates a settings group.
			 * Parent groups are created as necessary.
			 * @param[in] name The name of the group.
			 * @throw Exception Throws if the name is already in use.
			 */
			void createGroup(const std::string& name)
			{
				createAggregateSetting(name, libconfig::Setting::TypeGroup);
			}

			/**
			 * Creates a settings list.
			 * Parent groups are created as necessary.
			 * @param[in] name The name of the list.
			 * @throw Exception Throws if the name is already in use.
			 */
			void createList(const std::string& name)
			{
				createAggregateSetting(name, libconfig::Setting::TypeList);
			}

			/**
			 * Writes the configuration to the given path.
			 * @param[in] path The path to write to.
			 */
			void writeToFile(const std::string& path) const
			{
				config.writeFile(path.c_str());
			}

		public:
			/**
			 * The assignment operator.
			 *
			 * @param[in] rhs The instance to copy.
			 *
			 * @return Returns a reference to this instance.
			 */
			const Configuration& operator=(const Configuration& rhs)
			{
				ImplementationDetails::Configuration::copy(rhs.config, &config);
				return *this;
			}

		private:
			/**
			 * Creates an aggregate setting.
			 * Parent groups are created as necessary.
			 * @param[in] name The name of the group.
			 * @param[in] type The type of the setting.
			 * @throw Exception                Throws if the name is already in use.
			 * @throw InvalidArgumentException Throws if the type is not an aggregate setting type.
			 */
			void createAggregateSetting(const std::string& name, const libconfig::Setting::Type type)
			{
				if(has(name))
					OPENMPCD_THROW(Exception, "Name already in use.");

				if(
					type != libconfig::Setting::TypeGroup &&
					type != libconfig::Setting::TypeArray &&
					type != libconfig::Setting::TypeList)
				{
					OPENMPCD_THROW(InvalidArgumentException, "Invalid type given.");
				}

				std::vector<std::string> pathComponents;
				boost::algorithm::split(pathComponents, name, boost::algorithm::is_any_of("."));

				libconfig::Setting* currentSetting = &config.getRoot();
				for(unsigned int level=0; level < pathComponents.size(); ++level)
				{
					const std::string currentName = pathComponents[level];

					if(currentSetting->exists(currentName))
					{
						std::string lookupPath = currentSetting->getPath();
						if(!lookupPath.empty())
							lookupPath += ".";
						lookupPath += currentName;

						currentSetting = &config.lookup(lookupPath);
					}
					else
					{
						const bool isLastLevel = level + 1 == pathComponents.size();
						const libconfig::Setting::Type typeToCreate = isLastLevel ? type : libconfig::Setting::TypeGroup;
						currentSetting = &currentSetting->add(currentName, typeToCreate);
					}
				}
			}

		private:
			mutable libconfig::Config config; ///< The simulation configuration.
	};

	inline
	const Configuration::List
	Configuration::Setting::getList(const std::string& name) const
	{
		if(!has(name))
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				name + " does not exist.");
		}

		return List(setting->lookup(name));
	}

	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wsign-compare"
	#endif
	/**
	 * Sets a setting's value.
	 *
	 * @throw OpenMPCD::InvalidArgumentException
	 *        Throws if the setting already exists, but is of the wrong type.
	 *
	 * @param[in] setting
	 *            The name of the setting to set.
	 * @param[in] value
	 *            The value to set the setting to.
	 */
	template<> inline void Configuration::set(const std::string& setting, const unsigned int value)
	{
		if(static_cast<int>(value) == value)
		{
			set(setting, static_cast<int>(value));
		}
		else
		{
			#ifdef OPENMPCD_DEBUG
				if(static_cast<long long int>(value) != value)
					OPENMPCD_THROW(Exception, "Cannot represent config value.");
			#endif

			set(setting, static_cast<long long int>(value));
		}
	}
	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic pop
	#endif
}

#include <OpenMPCD/ImplementationDetails/Configuration/Configuration.hpp>

#endif
