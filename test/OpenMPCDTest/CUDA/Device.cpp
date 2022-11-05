/**
 * @file
 * Tests `OpenMPCD::CUDA::Device`.
 */

#include <OpenMPCD/CUDA/Device.hpp>

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/xpressive/xpressive.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::Device::getDeviceCount`",
	"[CUDA]")
{
	//the value 32 below is arbitrary, it was just chosen in such a way that
	//it catches absurdly high return values, e.g. 0xFF, without resulting in
	//false negatives for probably almost all systems
	REQUIRE(OpenMPCD::CUDA::Device::getDeviceCount() <= 32);
}


SCENARIO(
	"`OpenMPCD::CUDA::Device::getDomainID`",
	"[CUDA]")
{
	OpenMPCD::CUDA::Device device1;
	OpenMPCD::CUDA::Device device2;

	REQUIRE(device1.getPCIDomainID() == device1.getPCIDomainID());
	REQUIRE(device1.getPCIDomainID() == device2.getPCIDomainID());
}

SCENARIO(
	"`OpenMPCD::CUDA::Device::getPCIBusID`",
	"[CUDA]")
{
	OpenMPCD::CUDA::Device device1;
	OpenMPCD::CUDA::Device device2;

	REQUIRE(device1.getPCIBusID() == device1.getPCIBusID());
	REQUIRE(device1.getPCIBusID() == device2.getPCIBusID());
}

SCENARIO(
	"`OpenMPCD::CUDA::Device::getPCISlotID`",
	"[CUDA]")
{
	OpenMPCD::CUDA::Device device1;
	OpenMPCD::CUDA::Device device2;

	REQUIRE(device1.getPCISlotID() == device1.getPCISlotID());
	REQUIRE(device1.getPCISlotID() == device2.getPCISlotID());
}


SCENARIO(
	"`OpenMPCD::CUDA::Device::getPCIAddressString`",
	"[CUDA]")
{
	OpenMPCD::CUDA::Device device1;
	OpenMPCD::CUDA::Device device2;

	REQUIRE(device1.getPCIAddressString() == device1.getPCIAddressString());
	REQUIRE(device1.getPCIAddressString() == device2.getPCIAddressString());

	const std::string str = device1.getPCIAddressString();

	REQUIRE(str.length() == 4 + 1 + 2 + 1 + 2);

	const boost::xpressive::sregex regex =
		boost::xpressive::sregex::compile(
			"([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})");
	boost::xpressive::smatch match;

	REQUIRE(boost::xpressive::regex_match(str, match, regex));


	std::stringstream domain;
	domain << std::hex << std::setfill('0') << std::setw(4);
	domain << device1.getPCIDomainID();

	std::stringstream bus;
	bus << std::hex << std::setfill('0') << std::setw(2);
	bus << device1.getPCIBusID();

	std::stringstream slot;
	slot << std::hex << std::setfill('0') << std::setw(2);
	slot << device1.getPCISlotID();

	REQUIRE(match[1].str() == domain.str());
	REQUIRE(match[2].str() == bus.str());
	REQUIRE(match[3].str() == slot.str());
}


SCENARIO(
	"`OpenMPCD::CUDA::Device::getStackSizePerThread`, "
		"`OpenMPCD::CUDA::Device::setStackSizePerThread`",
	"[CUDA]")
{
	OpenMPCD::CUDA::Device device;

	const std::size_t oldSize = device.getStackSizePerThread();

	REQUIRE(oldSize > 0);


	REQUIRE_NOTHROW(device.setStackSizePerThread(oldSize));

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			device.setStackSizePerThread(0), OpenMPCD::InvalidArgumentException);
	#endif


	REQUIRE_NOTHROW(device.setStackSizePerThread(oldSize * 2));
	REQUIRE(device.getStackSizePerThread() == oldSize * 2);

	REQUIRE_NOTHROW(device.setStackSizePerThread(oldSize));
}
