#include <OpenMPCD/Utility/HostInformation.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/asio/ip/host_name.hpp>

#include <ctime>

namespace OpenMPCD
{
namespace Utility
{
namespace HostInformation
{

const std::string getHostname()
{
	return boost::asio::ip::host_name();
}

const std::string getCurrentUTCTimeAsString()
{
	static const std::size_t bufferSize = 32;

	const std::time_t t = std::time(0);

	char buffer[bufferSize + 1] = {0};

	const std::size_t ret =
		std::strftime(buffer, bufferSize, "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));

	if(ret == 0)
		OPENMPCD_THROW(Exception, "Call to `std::strftime` failed.");

	return buffer;
}

} //namespace HostInformation
} //namespace Utility
} //namespace OpenMPCD
