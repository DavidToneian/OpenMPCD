/**
 * @file
 * Tests functionality in `OpenMPCD::CUDA::BoundaryCondition::Factory`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/BoundaryCondition/Factory.hpp>

#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>

#include <boost/preprocessor/seq/for_each.hpp>

#include <iostream>

#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)

template<typename T>
static void test_getInstance()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::CUDA;
	using namespace OpenMPCD::CUDA::BoundaryCondition;

	static const unsigned int simBoxX = 5;
	static const unsigned int simBoxY = 5;
	static const unsigned int simBoxZ = 5;

	GIVEN("`boundaryConditions` without any settings")
	{
		Configuration config;
		config.createGroup("boundaryConditions");

		THEN("`getInstance` throws")
		{
			REQUIRE_THROWS_AS(
				Factory::getInstance(
					config.getSetting("boundaryConditions"),
					simBoxX, simBoxY, simBoxZ),
				InvalidConfigurationException);
		}
	}

	GIVEN("`boundaryConditions` with unknown boundary condition")
	{
		Configuration config;
		config.createGroup("boundaryConditions.unknown");

		THEN("`getInstance` throws")
		{
			REQUIRE_THROWS_AS(
				Factory::getInstance(
					config.getSetting("boundaryConditions"),
					simBoxX, simBoxY, simBoxZ),
				InvalidConfigurationException);
		}
	}

	GIVEN("`boundaryConditions` with two boundary conditions")
	{
		Configuration config;
		config.createGroup("boundaryConditions.LeesEdwards");
		config.createGroup("boundaryConditions.CrossGeometry");

		THEN("`getInstance` throws")
		{
			REQUIRE_THROWS_AS(
				Factory::getInstance(
					config.getSetting("boundaryConditions"),
					simBoxX, simBoxY, simBoxZ),
				InvalidConfigurationException);
		}
	}


	GIVEN("Lees-Edwards boundary conditions, old-style configuration")
	{
		static const FP shearRate = -1.23;
		Configuration config;
		config.set("boundaryConditions.LeesEdwards.shearRate", shearRate);

		THEN("`getInstance` works as expected")
		{
			//silence the warning about deprecated configuration style
			std::cout.setstate(std::ios_base::failbit);
			Base* const instance =
				Factory::getInstance(
					config.getSetting("boundaryConditions"),
					simBoxX, simBoxY, simBoxZ);
			std::cout.clear();

			REQUIRE(instance != NULL);

			LeesEdwards* const leesEdwards =
				dynamic_cast<LeesEdwards*>(instance);

			REQUIRE(leesEdwards != NULL);
			REQUIRE(leesEdwards->getShearRate() == shearRate);
		}
	}


	GIVEN("Lees-Edwards boundary conditions, new-style configuration")
	{
		static const FP shearRate = -1.23;
		Configuration config;
		config.set("boundaryConditions.LeesEdwards.shearRate", shearRate);

		THEN("`getInstance` works as expected")
		{
			Base* const instance =
				Factory::getInstance(
					config.getSetting("boundaryConditions"),
					simBoxX, simBoxY, simBoxZ);

			REQUIRE(instance != NULL);

			LeesEdwards* const leesEdwards =
				dynamic_cast<LeesEdwards*>(instance);

			REQUIRE(leesEdwards != NULL);
			REQUIRE(leesEdwards->getShearRate() == shearRate);
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::BoundaryCondition::Factory::getInstance`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getInstance<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}
