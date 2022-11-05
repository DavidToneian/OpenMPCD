/**
 * @file
 * This test runs a simulation with a `GaussianChains`, and checks whether the
 * results correspond to pre-recorded results.
 * The result, of course, depends on implementation details (e.g. the order in
 * which random variables are generated) that are a priori irrelevant to the
 * correctness of the implementation. However, this test is useful to detect
 * unintended deviations of known behavior. If, however, the implementation is
 * changed in a way that makes this test fail, that does not necessarily mean
 * that the new implementation is invalid.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Instrumentation.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/getGitCommitIdentifier.hpp>

#include <boost/filesystem.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>

#include <fstream>

static const OpenMPCD::Configuration getConfiguration()
{
	OpenMPCD::Configuration config;
	config.set("initialization.particleDensity", 10);
	config.set("initialization.particleVelocityDistribution.mean", 0.0);
	config.set("initialization.particleVelocityDistribution.standardDeviation", 1.0);

	config.set("mpc.simulationBoxSize.x", 2);
	config.set("mpc.simulationBoxSize.y", 5);
	config.set("mpc.simulationBoxSize.z", 1);
	config.set("mpc.timestep",            0.1);
	config.set("mpc.srdCollisionAngle",   2.27);
	config.set("mpc.gridShiftScale",      1.0);

	config.set("bulkThermostat.targetkT", 1.0);

	config.set("mpc.sweepSize", 10);

	config.createGroup("mpc.fluid.simple");

	config.set("bulkThermostat.type", "MBS");

	config.set("boundaryConditions.LeesEdwards.shearRate", 0.025);

	config.set("instrumentation.flowProfile.cellSubdivision.x", 1);
	config.set("instrumentation.flowProfile.cellSubdivision.y", 1);
	config.set("instrumentation.flowProfile.cellSubdivision.z", 1);

	return config;
}

static unsigned int getRNGSeed()
{
	return 12345;
}


static OpenMPCD::MPCParticleVelocityType getAnalyticShearFlowProfile(
	const OpenMPCD::MPCParticlePositionType y,
	const unsigned int simBoxSizeY,
	const OpenMPCD::FP shearRate,
	const unsigned int cellSubdivisionsY)
{
	using namespace OpenMPCD;

	MPCParticleVelocityType theoryY = y - 0.5 * simBoxSizeY;
		//make it so that the middle of the simulation system lies at y=0
	theoryY += 0.5 / cellSubdivisionsY;
		//make y denote the center of the cell considered
	return theoryY * shearRate;
}


template<typename T>
static bool twoSidedOneSampleTTestOfMean(
	const T sampleMean,
	const T sampleVariance,
	const std::size_t sampleSize,
	const T distributionMean,
	const T statisticalTestSignificanceLevel
	)
{
	typedef double U; //type for intermediate results

	const U t_statistic =
		(sampleMean - distributionMean) /
		sqrt(sampleVariance / U(sampleSize));

	boost::math::students_t studentTDist(sampleSize - 1);
	const U tmp =
		boost::math::cdf(
			boost::math::complement(studentTDist, fabs(t_statistic)));

	return tmp > statisticalTestSignificanceLevel / 2.0;
}


SCENARIO(
	"CUDA, Simple MPC Fluid, shear flow",
	"[CUDA]"
	)
{
	static unsigned int warmupSweepCount = 5000;
	static unsigned int measureSweepCount = 500;

	static const double singleTestSignificanceLevel = 0.2;
	static const double overallSignificanceLevel = 0.01;

	using namespace OpenMPCD;

	CUDA::Simulation simulation(getConfiguration(), getRNGSeed());
	CUDA::Instrumentation instrumentation(
		&simulation, getRNGSeed(), getGitCommitIdentifier());
	const CUDA::MPCFluid::Base& fluid = simulation.getMPCFluid();

	const std::size_t particleCount = fluid.getParticleCount();
	const std::size_t coordinateCount = 3 * particleCount;

	for(unsigned int iteration = 0; iteration < warmupSweepCount; ++iteration)
	{
		simulation.sweep();
	}


	for(unsigned int iteration = 0; iteration < measureSweepCount; ++iteration)
	{
		simulation.sweep();
		instrumentation.measure();
	}


	const boost::filesystem::path tempdir =
		boost::filesystem::temp_directory_path() /
		boost::filesystem::unique_path();
	boost::filesystem::create_directories(tempdir);

	instrumentation.save(tempdir.c_str());


	const OpenMPCD::Configuration& config = simulation.getConfiguration();

	const double shearRate =
		config.read<double>("boundaryConditions.LeesEdwards.shearRate");
	const unsigned int cellSubdivisionY =
		config.read<unsigned int>(
			"instrumentation.flowProfile.cellSubdivision.y");


	/* In a given volume's center-of-mass frame, the Cartesian particle
	   velocity components are normally distributed with mean \f$ 0 \f$ and
	   variance \f$ \sigma_N^2 = k_B T / m \f$, \f$ k_B \f$ being
	   Boltzmann's constant, \f$ T \f$ being the temperature, and \f$ m \f$
	   being the particle's mass [1].
	   In the case of the shear flow direction (which is `x` here), another
	   term is added to the particle's velocity component in the
	   center-of-mass frame of the given volume, which is
	   \f$ U = \dot{\gamma} Y \f$, with \f$ \dot{\gamma} \f$ being the shear
	   rate, and \f$ Y \f$ denoting the random position in the given volume
	   along the gradient direction (here, this is the `y` direction), with
	   `0` being the center of the given volume along that direction;
	   \f$ Y \f$ is thus distributed according to the uniform distribution
	   over the range \f$ \left[ -h/2, h/2 \right) \f$. With the MPC
	   collision cell size being set to \f$ 1 \f$, and if \f$ c \f$ being
	   the number of times the MPC collision cell is sub-divided for
	   sampling along the gradient direction, one has \f$ h = 1 / c \f$.
	   Since \f$ Y \f$ is independent from the normally-distributed
	   contribution to the particle's Cartesian velocity component, the
	   variance \f$ \sigma_{v_x}^2 \f$ of that component is given as the
	   sum of the variance of the normally-distributed contribution,
	   \f$ \sigma_N^2 \f$, and the variance of \f$ Y \f$,
	   \f$ \sigma_U^2 = \dot{\gamma}^2 h^2 / 12 \f$ [2], so that
	   \f$ \sigma_{v_x}^2 = k_B T / m + \dot{\gamma}^2 h^2 / 12 \f$ and
	   \f$ \sigma_{v_i}^2 = k_B T / m \f$ for the other Cartesian
	   velocity components \f$ v_i, i \in \left\{ y, z \right\} \f$.

	   The sample variance
	   \f$ S_n^2 =
		   \frac{ 1 }{ n - 1 }
		   \sum_{j = 1}^n \left( {v_i}_j - \mu \right) \f$
	   of the random sample \f$ {v_i}_j, j \in \left[ 1, n \right] \f$ of
	   Cartesian velocity coordinates \f$ v_i \f$ has an expectation value
	   of \f$ \sigma_{v_i}^2 \f$ and a variance of
	   \f$ n^{-1}
		   \left( \mu_{4,i} - \frac{ n - 3 }{ n - 1 } \sigma_{v_i}^4
		   \right) \f$
	   and is, for large sample sizes \f$ n \f$, approximately normally
	   distributed [3-7]. Here, \f$ \mu_{4,i} \f$ is the fourth central
	   moment of the distribution of Cartesian velocity coordinates in the
	   direction \f$ i \in \left\{ x, y, z \right\} \f$, which is related to
	   the kurtosis \f$ \ķappa_i \f$ via
	   \f$ \kappa_i = \mu_{4,i} / \sigma_{v_i}^4 \f$ [8].
	   With \f$ \kappa_{e,i} = \kappa_i - 3 \f$ being the excess curtosis,
	   one can calculate [8] its values for the Cartesian velocity
	   coordinates, which are the sum of up to two independently distributed
	   numbers, from the excess kurtosis of those two underlying
	   distributions to be
	   \f$ \kappa_{e,x} = \left( \sigma_N^2 + \sigma_U^2 \right)^{-2}
	   \sigma_U^4 \cdot \left( - 6 / 5 \right) \f$,
	   \f$ \kappa_{e,y} = \kappa_{e,z} = 0 \f$,
	   using the fact that the excess kurtosis of the normal distribution
	   is \f$ 0 \f$ [9] and that the excess kurtosis of the uniform
	   distribution is \f$ - 6 / 5 \f$ [2].



	   References:
	   [1] https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#Derivation_and_related_distributions
	   [2] https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
	   [3] A. M. Mood, F. A. Graybill, D. C. Boes:
		   Introduction to the Theory of Statistics, 3rd edition (1974),
		   McGraw-Hill. Page 229.
	   [4] https://math.stackexchange.com/a/73080/234597
	   [5] https://stats.stackexchange.com/a/29945
	   [6] https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance
	   [7] https://stats.stackexchange.com/a/29920
	   [8] https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments
	   [9] https://en.wikipedia.org/wiki/Normal_distribution
	 */

	const double kT = config.read<double>("bulkThermostat.targetkT");
	const double m = fluid.getParticleMass();

	const double sigma_N_squared = kT / m;

	const double uniformDistributionLength = 1.0 / cellSubdivisionY;

	const double sigma_U_squared =
		pow(shearRate * uniformDistributionLength, 2) / 12.0;

	const double velocityVariances[3] =
		{ sigma_N_squared + sigma_U_squared,
		  sigma_N_squared, sigma_N_squared };

	const double excessKurtosisX =
		pow(sigma_U_squared, 2) * (-6.0 / 5.0) /
		pow(sigma_N_squared + sigma_U_squared, 2);

	const double kurtosisX = excessKurtosisX + 3;
	const double kurtosisYZ = 3;
	const double fourthCentralMoments[3] =
		{ velocityVariances[0] * velocityVariances[0] * kurtosisX,
		  velocityVariances[1] * velocityVariances[1] * kurtosisYZ,
		  velocityVariances[2] * velocityVariances[2] * kurtosisYZ
		};




	std::ifstream file((tempdir / "flowProfile.data").c_str());
	std::string header;
	std::getline(file, header);

	MPCParticlePositionType posX, posY, posZ;
	MPCParticleVelocityType velX, velY, velZ;
	MPCParticleVelocityType stdVelX, stdVelY, stdVelZ;
	unsigned long int sampleSize;

	std::size_t failedStatisticalTests = 0;

	unsigned int comparisonCount = 0;
	while(
		file
		>> posX >> posY >> posZ
		>> velX >> velY >> velZ
		>> stdVelX >> stdVelY >> stdVelZ
		>> sampleSize)
	{
		const MPCParticleVelocityType sampleMeans[3] =
			{velX, velY, velZ};
		const MPCParticleVelocityType sampleStandardDeviations[3] =
			{stdVelX, stdVelY, stdVelZ};

		//test means
		for(std::size_t i = 0; i < 3; ++i)
		{
			const MPCParticleVelocityType sampleVariance =
				sampleStandardDeviations[i] * sampleStandardDeviations[i];
			MPCParticleVelocityType distributionMean = 0;
			if(i == 0)
			{
				distributionMean =
					getAnalyticShearFlowProfile(
						posY, simulation.getSimulationBoxSizeY(),
						shearRate, cellSubdivisionY);
			}

			const bool testResult =
				twoSidedOneSampleTTestOfMean(
					sampleMeans[i],
					sampleVariance,
					sampleSize,
					distributionMean,
					singleTestSignificanceLevel);

			if(!testResult)
				++failedStatisticalTests;
		}


		REQUIRE(stdVelX >= 0);
		REQUIRE(stdVelY >= 0);
		REQUIRE(stdVelZ >= 0);

		//test variances
		for(std::size_t i = 0; i < 3; ++i)
		{
			const MPCParticleVelocityType sampleVariance =
				sampleStandardDeviations[i] * sampleStandardDeviations[i];

			//We don't have access to the data points anymore, so instead of
			//calculating the true sample variance of sample variances, we
			//calculate its expectation value:
			const MPCParticleVelocityType expectedVarianceOfSampleVariance =
				(
					fourthCentralMoments[0] -
					(sampleSize - 3.0) / (sampleSize - 1.0) *
					velocityVariances[i] * velocityVariances[i]
				) / MPCParticleVelocityType(sampleSize);


			//Then, we approximate what the sample variance of sample variances
			//would have been:
			const MPCParticleVelocityType fakedSampleVarianceOfSampleVariance =
				expectedVarianceOfSampleVariance * sampleSize;

			const bool testResult =
				twoSidedOneSampleTTestOfMean(
					sampleVariance,
					fakedSampleVarianceOfSampleVariance,
					sampleSize,
					velocityVariances[i],
					singleTestSignificanceLevel);

			if(!testResult)
				++failedStatisticalTests;
		}

		++comparisonCount;
	}

	const unsigned int cellCount =
		simulation.getSimulationBoxSizeX() *
		simulation.getSimulationBoxSizeY() *
		simulation.getSimulationBoxSizeZ();
	REQUIRE(comparisonCount == cellCount);


	const std::size_t totalTestCount = 2 * 3 * comparisonCount;


	boost::math::binomial binomialDistribution(
		totalTestCount, singleTestSignificanceLevel);
	const double lowerEndOfProjectionInterval =
		boost::math::quantile(
			binomialDistribution, overallSignificanceLevel / 2.0);
	const double upperEndOfProjectionInterval =
		boost::math::quantile(boost::math::complement(
			binomialDistribution, overallSignificanceLevel / 2.0));

	REQUIRE(lowerEndOfProjectionInterval >= 1);
	REQUIRE(upperEndOfProjectionInterval <= totalTestCount - 1);

	REQUIRE(failedStatisticalTests >= lowerEndOfProjectionInterval);
	REQUIRE(failedStatisticalTests <= upperEndOfProjectionInterval);

	/*
	const double probabilityOfFailureLower =
		boost::math::cdf(
			binomialDistribution, lowerEndOfProjectionInterval - 1);
	const double probabilityOfFailureUpper =
		1 -
		boost::math::cdf(
			binomialDistribution, upperEndOfProjectionInterval);
	const double probabilityOfFailure =
		probabilityOfFailureLower + probabilityOfFailureUpper;
	*/


	boost::filesystem::remove_all(tempdir);
}
