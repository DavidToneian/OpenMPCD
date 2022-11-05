/**
 * @file
 * Tests functionality in
 * `OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCSolute/ImplementationDetails/StarPolymers.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <boost/preprocessor/seq/for_each.hpp>

#include <set>

#define ITERATE_CONFIGURATION_VARIABLES \
	for(std::size_t starCount = 0; starCount <= 2; ++starCount) \
	for(std::size_t armsPerStar = 2; armsPerStar <= 3; ++armsPerStar) \
	for(std::size_t partsPerArm = 4; partsPerArm <= 5; ++partsPerArm) \
	for(int hasMagnetic = 0; hasMagnetic <= 1; ++hasMagnetic)


SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"getParticleCountPerStar`",
	"[CUDA]")
{
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	ITERATE_CONFIGURATION_VARIABLES
	{
		if(starCount != 1)
			continue;

		const std::size_t expected =
			1 + (partsPerArm + hasMagnetic) * armsPerStar;
		REQUIRE(
			getParticleCountPerStar(
				armsPerStar, partsPerArm,
				static_cast<bool>(hasMagnetic))
			== expected);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"getParticleCount`",
	"[CUDA]")
{
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	ITERATE_CONFIGURATION_VARIABLES
	{
		const std::size_t expectedPerStar =
			1 + (partsPerArm + hasMagnetic) * armsPerStar;
		REQUIRE(
			getParticleCount(
				starCount,
				armsPerStar, partsPerArm,
				static_cast<bool>(hasMagnetic))
			== expectedPerStar * starCount);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"getParticleStructureIndices`",
	"[CUDA]")
{
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	ITERATE_CONFIGURATION_VARIABLES
	{
		const bool magnetic = static_cast<bool>(hasMagnetic);
		const std::size_t particleCount =
			getParticleCount(
				starCount, armsPerStar, partsPerArm, magnetic);
		const std::size_t particleCountPerStar =
			getParticleCountPerStar(
				armsPerStar, partsPerArm, magnetic);
		const std::size_t totalParticlesPerArm = partsPerArm + hasMagnetic;

		for(std::size_t partID = 0; partID < particleCount; ++partID)
		{
			std::size_t p = partID;

			std::size_t expectedStarID = 0;
			while(p >= particleCountPerStar)
			{
				p -= particleCountPerStar;
				++expectedStarID;
			}

			const bool expectedIsCore = p == 0;

			if(!expectedIsCore)
				--p;

			std::size_t expectedArmID = 0;
			while(p >= totalParticlesPerArm)
			{
				p -= totalParticlesPerArm;
				++expectedArmID;
			}

			const bool expectedIsMagnetic = p == partsPerArm;

			const std::size_t expectedParticleIDInArm = p;

			std::size_t starID;
			bool isCoreParticle;
			std::size_t armID = 99999;
			bool isMagneticParticle;
			std::size_t particleIDInArm = 99999;

			getParticleStructureIndices(
				partID,
				starCount, armsPerStar, partsPerArm, magnetic,
				&starID, &isCoreParticle, &armID, &isMagneticParticle,
				&particleIDInArm);


			REQUIRE(starID == expectedStarID);
			REQUIRE(isCoreParticle == expectedIsCore);

			if(isCoreParticle)
			{
				REQUIRE(armID == 99999);
			}
			else
			{
				REQUIRE(armID == expectedArmID);
			}

			REQUIRE(isMagneticParticle == expectedIsMagnetic);

			if(isCoreParticle)
			{
				REQUIRE(particleIDInArm == 99999);
			}
			else
			{
				if(isMagneticParticle)
				{
					REQUIRE(particleIDInArm == partsPerArm);
				}
				else
				{
					REQUIRE(particleIDInArm == expectedParticleIDInArm);
				}
			}
		}
	}

	for(std::size_t starCount = 0; starCount <= 3; ++starCount)
	{
		for(std::size_t armsPerStar = 0; armsPerStar <= 3; ++armsPerStar)
		{
			for(std::size_t partsPerArm = 0; partsPerArm <= 5; ++partsPerArm)
			{
				for(int hasMagnetic = 0; hasMagnetic <= 1; ++hasMagnetic)
				{
					const std::size_t expectedPerStar =
						1 + (partsPerArm + hasMagnetic) * armsPerStar;
					REQUIRE(
						getParticleCount(
							starCount,
							armsPerStar, partsPerArm,
							static_cast<bool>(hasMagnetic))
						== expectedPerStar * starCount);
				}
			}
		}
	}

	#ifdef OPENMPCD_DEBUG
		{
			std::size_t starID;
			bool isCoreParticle;
			std::size_t armID;
			bool isMagneticParticle;
			std::size_t particleIDInArm ;

			REQUIRE_THROWS_AS(
				getParticleStructureIndices(
					0,
					1, 1, 1, true,
					NULL, &isCoreParticle, &armID, &isMagneticParticle,
					&particleIDInArm),
				OpenMPCD::NULLPointerException);

			REQUIRE_THROWS_AS(
				getParticleStructureIndices(
					0,
					1, 1, 1, true,
					&starID, NULL, &armID, &isMagneticParticle,
					&particleIDInArm),
				OpenMPCD::NULLPointerException);

			REQUIRE_THROWS_AS(
				getParticleStructureIndices(
					0,
					1, 1, 1, true,
					&starID, &isCoreParticle, NULL, &isMagneticParticle,
					&particleIDInArm),
				OpenMPCD::NULLPointerException);

			REQUIRE_THROWS_AS(
				getParticleStructureIndices(
					0,
					1, 1, 1, true,
					&starID, &isCoreParticle, &armID, NULL,
					&particleIDInArm),
				OpenMPCD::NULLPointerException);

			REQUIRE_THROWS_AS(
				getParticleStructureIndices(
					0,
					1, 1, 1, true,
					&starID, &isCoreParticle, &armID, &isMagneticParticle,
					NULL),
				OpenMPCD::NULLPointerException);

			REQUIRE_THROWS_AS(
				getParticleStructureIndices(
					3,
					1, 1, 1, true,
					&starID, &isCoreParticle, &armID, &isMagneticParticle,
					&particleIDInArm),
				OpenMPCD::InvalidArgumentException);
		}
	#endif
}

SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"getParticleType`",
	"[CUDA]")
{
	using namespace OpenMPCD::CUDA;
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	ITERATE_CONFIGURATION_VARIABLES
	{
		const bool magnetic = static_cast<bool>(hasMagnetic);
		const std::size_t particleCount =
			getParticleCount(
				starCount, armsPerStar, partsPerArm, magnetic);
		const std::size_t particleCountPerStar =
			getParticleCountPerStar(
				armsPerStar, partsPerArm, magnetic);
		const std::size_t totalParticlesPerArm = partsPerArm + hasMagnetic;

		for(std::size_t partID = 0; partID < particleCount; ++partID)
		{
			std::size_t p = partID;

			std::size_t expectedStarID = 0;
			while(p >= particleCountPerStar)
			{
				p -= particleCountPerStar;
				++expectedStarID;
			}

			const bool expectedIsCore = p == 0;

			if(!expectedIsCore)
				--p;

			std::size_t expectedArmID = 0;
			while(p >= totalParticlesPerArm)
			{
				p -= totalParticlesPerArm;
				++expectedArmID;
			}

			const bool expectedIsMagnetic = p == partsPerArm;

			const typename ParticleType::Enum particleType =
				getParticleType(
					partID,
					starCount, armsPerStar, partsPerArm, magnetic);

			if(expectedIsCore)
			{
				REQUIRE(particleType == ParticleType::Core);
			}
			else
			{
				if(expectedIsMagnetic)
				{
					REQUIRE(particleType == ParticleType::Magnetic);
				}
				else
				{
					REQUIRE(particleType == ParticleType::Arm);
				}
			}
		}
	}

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			getParticleType(
				3,
				1, 1, 1, true),
			OpenMPCD::InvalidArgumentException);
	#endif
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"particlesAreBonded`",
	"[CUDA]")
{
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	ITERATE_CONFIGURATION_VARIABLES
	{
		const bool magnetic = static_cast<bool>(hasMagnetic);
		const std::size_t particleCount =
			getParticleCount(
				starCount, armsPerStar, partsPerArm, magnetic);
		const std::size_t particleCountPerStar =
			getParticleCountPerStar(
				armsPerStar, partsPerArm, magnetic);

		for(std::size_t p1 = 0; p1 < particleCount; ++p1)
		{
			std::size_t starID1;
			bool isCoreParticle1;
			std::size_t armID1;
			bool isMagneticParticle1;
			std::size_t particleIDInArm1;

			getParticleStructureIndices(
				p1,
				starCount, armsPerStar, partsPerArm, magnetic,
				&starID1, &isCoreParticle1, &armID1, &isMagneticParticle1,
				&particleIDInArm1);

			for(std::size_t p2 = 0; p2 < particleCount; ++p2)
			{
				std::size_t starID2;
				bool isCoreParticle2;
				std::size_t armID2;
				bool isMagneticParticle2;
				std::size_t particleIDInArm2;

				getParticleStructureIndices(
					p2,
					starCount, armsPerStar, partsPerArm, magnetic,
					&starID2, &isCoreParticle2, &armID2, &isMagneticParticle2,
					&particleIDInArm2);

				if(p1 == p2)
				{
					#ifdef OPENMPCD_DEBUG
						REQUIRE_THROWS_AS(
							particlesAreBonded(
								p1, p2,
								starCount, armsPerStar, partsPerArm, magnetic),
							OpenMPCD::InvalidArgumentException);
					#endif

					continue;
				}

				const bool bonded = particlesAreBonded(
					p1, p2, starCount, armsPerStar, partsPerArm, magnetic);

				if(starID1 != starID2)
				{
					REQUIRE_FALSE(bonded);
					continue;
				}

				if(isCoreParticle1)
				{
					if(isCoreParticle2)
					{
						REQUIRE_FALSE(bonded);
						continue;
					}

					const bool expected = particleIDInArm2 == 0;
					REQUIRE(bonded == expected);
					continue;
				}

				if(isCoreParticle2)
				{
					const bool expected = particleIDInArm1 == 0;
					REQUIRE(bonded == expected);
					continue;
				}


				if(armID1 != armID2)
				{
					REQUIRE_FALSE(bonded);
					continue;
				}

				bool expected = false;
				if(particleIDInArm1 + 1 == particleIDInArm2)
					expected = true;
				if(particleIDInArm1 == particleIDInArm2 + 1)
					expected = true;

				REQUIRE(bonded == expected);
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"getParticleTypeCombinationIndex`",
	"[CUDA]")
{
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	static const std::size_t typeCount = 3;

	std::set<std::size_t> seenIndices;

	for(unsigned int t1 = 0; t1 < typeCount; ++t1)
	{
		for(unsigned int t2 = t1; t2 < typeCount; ++t2)
		{
			const ParticleType::Enum type1 =
				static_cast<ParticleType::Enum>(t1);
			const ParticleType::Enum type2 =
				static_cast<ParticleType::Enum>(t2);

			const std::size_t index =
				getParticleTypeCombinationIndex(type1, type2);

			REQUIRE(index == getParticleTypeCombinationIndex(type2, type1));

			REQUIRE(seenIndices.count(index) == 0);

			seenIndices.insert(index);
		}
	}

	for(std::size_t index = 0; index < 6; ++index)
	{
		REQUIRE(seenIndices.count(index) != 0);
	}

	REQUIRE(seenIndices.size() == 6);
}

template<typename T>
__global__
void createInteractionsOnDevice_destroyInteractionsOnDevice_test_kernel(
	OpenMPCD::PairPotentials::
		WeeksChandlerAndersen_DistanceOffset<T>** const
			wca,
	OpenMPCD::PairPotentials::FENE<T>** const fene,
	OpenMPCD::PairPotentials::
		MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>** const
			magnetic,
	bool* const testPassed
	)
{
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	OPENMPCD_DEBUG_ASSERT(testPassed != NULL);

	*testPassed = true;

	#define REQUIRE_DEVICE(condition) \
		if(!(condition)) \
		{ \
			printf("failed assert: %s\n", #condition); \
			*testPassed = false; \
			return; \
		}

	REQUIRE_DEVICE(wca != NULL);
	REQUIRE_DEVICE(fene != NULL);
	REQUIRE_DEVICE(magnetic != NULL);

	//check that all WCA potentials are different and non-NULL
	for(std::size_t i = 0; i < 6; ++i)
	{
		REQUIRE_DEVICE(wca[i] != NULL);

		for(std::size_t j = i + 1; j < 6; ++j)
		{
			REQUIRE_DEVICE(wca[i] != wca[j]);
		}
	}

	const std::size_t typeComboIndexCC =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Core);
	const std::size_t typeComboIndexCA =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Arm);
	const std::size_t typeComboIndexCM =
		getParticleTypeCombinationIndex(
			ParticleType::Core, ParticleType::Magnetic);
	const std::size_t typeComboIndexAA =
		getParticleTypeCombinationIndex(
			ParticleType::Arm, ParticleType::Arm);
	const std::size_t typeComboIndexAM =
		getParticleTypeCombinationIndex(
			ParticleType::Arm, ParticleType::Magnetic);
	const std::size_t typeComboIndexMM =
		getParticleTypeCombinationIndex(
			ParticleType::Magnetic, ParticleType::Magnetic);

	REQUIRE_DEVICE(fene[typeComboIndexCA] != NULL);
	REQUIRE_DEVICE(fene[typeComboIndexAA] != NULL);
	REQUIRE_DEVICE(fene[typeComboIndexAM] != NULL);

	REQUIRE_DEVICE(fene[typeComboIndexCA] != fene[typeComboIndexAA]);
	REQUIRE_DEVICE(fene[typeComboIndexCA] != fene[typeComboIndexAM]);
	REQUIRE_DEVICE(fene[typeComboIndexAA] != fene[typeComboIndexAM]);

	using namespace OpenMPCD::PairPotentials;

	static const T epsilon_C = 1;
	static const T epsilon_A = 2;
	static const T epsilon_M = 3;

	static const T sigma_C = 4;
	static const T sigma_A = 5;
	static const T sigma_M = 6;

	static const T D_C = 7;
	static const T D_A = 8;
	static const T D_M = 9;

	static const T magneticPrefactor = 10;
	OpenMPCD::Vector3D<T> dipoleOrientation(1, 2, 3);
	dipoleOrientation.normalize();

	const T epsilon_CC = epsilon_C;
	const T epsilon_CA = sqrt( epsilon_C * epsilon_A );
	const T epsilon_CM = sqrt( epsilon_C * epsilon_M );
	const T epsilon_AA = epsilon_A;
	const T epsilon_AM = sqrt( epsilon_A * epsilon_M);
	const T epsilon_MM = epsilon_M;

	const T sigma_CC = sigma_C;
	const T sigma_CA = 0.5 * (sigma_C + sigma_A);
	const T sigma_CM = 0.5 * (sigma_C + sigma_M);
	const T sigma_AA = sigma_A;
	const T sigma_AM = 0.5 * (sigma_A + sigma_M);
	const T sigma_MM = sigma_M;

	const T D_CC = D_C;
	const T D_CA = 0.5 * (D_C + D_A);
	const T D_CM = 0.5 * (D_C + D_M);
	const T D_AA = D_A;
	const T D_AM = 0.5 * (D_A + D_M);
	const T D_MM = D_M;

	const T K_CA = 30 * epsilon_CA / (sigma_CA * sigma_CA);
	const T K_AA = 30 * epsilon_AA / (sigma_AA * sigma_AA);
	const T K_AM = 30 * epsilon_AM / (sigma_AM * sigma_AM);

	const T R_CA = 1.5 * sigma_CA;
	const T R_AA = 1.5 * sigma_AA;
	const T R_AM = 1.5 * sigma_AM;

	const T l_0_CA = D_CA;
	const T l_0_AA = D_AA;
	const T l_0_AM = D_AM;


	REQUIRE_DEVICE(wca[typeComboIndexCC]->getEpsilon() == epsilon_CC);
	REQUIRE_DEVICE(wca[typeComboIndexCC]->getSigma() == sigma_CC);
	REQUIRE_DEVICE(wca[typeComboIndexCC]->getD() == D_CC);

	REQUIRE_DEVICE(wca[typeComboIndexCA]->getEpsilon() == epsilon_CA);
	REQUIRE_DEVICE(wca[typeComboIndexCA]->getSigma() == sigma_CA);
	REQUIRE_DEVICE(wca[typeComboIndexCA]->getD() == D_CA);

	REQUIRE_DEVICE(wca[typeComboIndexCM]->getEpsilon() == epsilon_CM);
	REQUIRE_DEVICE(wca[typeComboIndexCM]->getSigma() == sigma_CM);
	REQUIRE_DEVICE(wca[typeComboIndexCM]->getD() == D_CM);

	REQUIRE_DEVICE(wca[typeComboIndexAA]->getEpsilon() == epsilon_AA);
	REQUIRE_DEVICE(wca[typeComboIndexAA]->getSigma() == sigma_AA);
	REQUIRE_DEVICE(wca[typeComboIndexAA]->getD() == D_AA);

	REQUIRE_DEVICE(wca[typeComboIndexAM]->getEpsilon() == epsilon_AM);
	REQUIRE_DEVICE(wca[typeComboIndexAM]->getSigma() == sigma_AM);
	REQUIRE_DEVICE(wca[typeComboIndexAM]->getD() == D_AM);

	REQUIRE_DEVICE(wca[typeComboIndexMM]->getEpsilon() == epsilon_MM);
	REQUIRE_DEVICE(wca[typeComboIndexMM]->getSigma() == sigma_MM);
	REQUIRE_DEVICE(wca[typeComboIndexMM]->getD() == D_MM);


	REQUIRE_DEVICE(fene[typeComboIndexCA]->getK() == K_CA);
	REQUIRE_DEVICE(fene[typeComboIndexCA]->getR() == R_CA);
	REQUIRE_DEVICE(fene[typeComboIndexCA]->get_l_0() == l_0_CA);

	if(boost::is_same<T, float>::value)
	{
		REQUIRE_DEVICE(abs(fene[typeComboIndexAA]->getK() - K_AA) < 1e-6);
	}
	else
	{
		REQUIRE_DEVICE(fene[typeComboIndexAA]->getK() == K_AA);
	}
	REQUIRE_DEVICE(fene[typeComboIndexAA]->getR() == R_AA);
	REQUIRE_DEVICE(fene[typeComboIndexAA]->get_l_0() == l_0_AA);

	REQUIRE_DEVICE(fene[typeComboIndexAM]->getK() == K_AM);
	REQUIRE_DEVICE(fene[typeComboIndexAM]->getR() == R_AM);
	REQUIRE_DEVICE(fene[typeComboIndexAM]->get_l_0() == l_0_AM);


	REQUIRE_DEVICE((*magnetic)->getPrefactor() == magneticPrefactor);
	REQUIRE_DEVICE((*magnetic)->getDipoleOrientation() == dipoleOrientation);

	#undef REQUIRE_DEVICE
}
template<typename T>
void createInteractionsOnDevice_destroyInteractionsOnDevice_test()
{
	using namespace OpenMPCD::PairPotentials;
	using namespace
		OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers;

	typedef WeeksChandlerAndersen_DistanceOffset<T> WCA;
	typedef FENE<T> FENE;
	typedef MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		Magnetic;

	WCA** wcaPotentials = NULL;
	FENE** fenePotentials = NULL;
	Magnetic** magneticPotential = NULL;

	static const T epsilon_core = 1;
	static const T epsilon_arm = 2;
	static const T epsilon_magnetic = 3;

	static const T sigma_core = 4;
	static const T sigma_arm = 5;
	static const T sigma_magnetic = 6;

	static const T D_core = 7;
	static const T D_arm = 8;
	static const T D_magnetic = 9;

	static const T magneticPrefactor = 10;
	OpenMPCD::Vector3D<T> dipoleOrientation(1, 2, 3);
	dipoleOrientation.normalize();

	createInteractionsOnDevice(
		epsilon_core, epsilon_arm, epsilon_magnetic,
		sigma_core, sigma_arm, sigma_magnetic,
		D_core, D_arm, D_magnetic,
		magneticPrefactor, dipoleOrientation,
		&wcaPotentials, &fenePotentials, &magneticPotential
		);

	REQUIRE(wcaPotentials != NULL);
	REQUIRE(fenePotentials != NULL);
	REQUIRE(magneticPotential != NULL);

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	REQUIRE(dmm.isDeviceMemoryPointer(wcaPotentials));
	REQUIRE(dmm.isDeviceMemoryPointer(fenePotentials));
	REQUIRE(dmm.isDeviceMemoryPointer(magneticPotential));

	bool* d_passed;
	dmm.allocateMemory(&d_passed, 1);

	createInteractionsOnDevice_destroyInteractionsOnDevice_test_kernel
		<<<1, 1>>>(wcaPotentials, fenePotentials, magneticPotential, d_passed);

	bool true_ = true;
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, d_passed, 1));

	destroyInteractionsOnDevice(
		wcaPotentials, fenePotentials, magneticPotential);
}
SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"createInteractionsOnDevice`, "
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"destroyInteractionsOnDevice`, ",
	"[CUDA]")
{
	#define SEQ_DATATYPE (double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		createInteractionsOnDevice_destroyInteractionsOnDevice_test<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::"
	"computeForceOnParticle1DueToParticle2`",
	"[CUDA]")
{
	WARN("test missing");
}

