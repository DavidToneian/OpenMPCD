#include <OpenMPCD/Configuration.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Utility/CompilerDetection.hpp>

#include <limits>

namespace OpenMPCD
{

bool Configuration::Setting::childrenHaveNamesInCollection(
	const std::set<std::string>& names,
	std::string* const offender) const
{
	for(std::size_t i = 0; i < getChildCount(); ++i)
	{
		if(names.count(getChild(i).getName()) == 0)
		{
			if(offender)
				*offender = getChild(i).getName();
			return false;
		}
	}

	return true;
}

} //namespace OpenMPCD
