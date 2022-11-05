/**
 * @file
 * Tests functionality in `OpenMPCD::CUDA::BoundaryCondition::LeesEdwards`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>


SCENARIO(
	"`OpenMPCD::CUDA::BoundaryCondition::LeesEdwards::getTotalAvailableVolume`",
	"")
{
	static const unsigned int systemSizes[][3] =
	{
		{0, 0, 0},
		{0, 4, 5},
		{3, 0, 5},
		{3, 4, 0},
		{3, 4, 5},
		{10, 10, 10},
		{10, 20, 30}
	};
	static const std::size_t systemSizesCount =
		sizeof(systemSizes) / sizeof(systemSizes[0]);

	GIVEN("A valid instance")
	{
		OpenMPCD::Configuration config;
		config.set("boundaryConditions.LeesEdwards.shearRate", 0.123);

		for(std::size_t ssi = 0; ssi < systemSizesCount; ++ssi)
		{
			const unsigned int L_x = systemSizes[ssi][0];
			const unsigned int L_y = systemSizes[ssi][1];
			const unsigned int L_z = systemSizes[ssi][2];

			OpenMPCD::CUDA::BoundaryCondition::LeesEdwards lebc(
				config.getSetting("boundaryConditions.LeesEdwards"),
				L_x, L_y, L_z);

			REQUIRE(lebc.getTotalAvailableVolume() == L_x * L_y * L_z);
		}
	}
}
