#include <OpenMPCD/AnalyticQuantities.hpp>

#include <boost/math/constants/constants.hpp>
#include <cmath>

using namespace OpenMPCD;

static const FP pi = boost::math::constants::pi<FP>();

FP AnalyticQuantities::meanFreePath(const FP kT, const FP m, const FP timestep)
{
	return timestep*sqrt(kT/m);
}

FP AnalyticQuantities::kineticContributionsToSRDKinematicShearViscosity(
		const FP kT, const FP m, const FP meanParticleCountPerCell, const FP srdAngle, const FP timestep)
{
	const FP M = meanParticleCountPerCell;

	const FP factor1      = kT*timestep/(2*m);
	const FP denominator1 = M - 1 + exp(-M);
	const FP denominator2 = 2 - cos(srdAngle) - cos(2*srdAngle);
	const FP factor2      = 5*M / (denominator1 * denominator2) - 1;

	return factor1 * factor2;
}

FP AnalyticQuantities::kineticContributionsToSRDDynamicShearViscosity(
		const FP kT, const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,
		const FP srdAngle, const FP timestep)
{
	const FP cellVolume      = pow(linearCellSize, 3);
	const FP particleDensity = meanParticleCountPerCell / cellVolume;
	const FP massDensity     = particleDensity * m;

	return massDensity * kineticContributionsToSRDKinematicShearViscosity(
			kT, m, meanParticleCountPerCell, srdAngle, timestep);
}

FP AnalyticQuantities::collisionalContributionsToSRDKinematicShearViscosity(
		const FP linearCellSize, const FP meanParticleCountPerCell, const FP srdAngle, const FP timestep)
{
	const FP M = meanParticleCountPerCell;

	const FP factor1 = linearCellSize * linearCellSize / (6*3*timestep);
	const FP factor2 = (M - 1 + exp(-M)) / M;
	const FP factor3 = 1 - cos(srdAngle);

	return factor1 * factor2 * factor3;
}

FP AnalyticQuantities::collisionalContributionsToSRDDynamicShearViscosity(
		const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,	const FP srdAngle,
		const FP timestep)
{
	const FP cellVolume      = pow(linearCellSize, 3);
	const FP particleDensity = meanParticleCountPerCell / cellVolume;
	const FP massDensity     = particleDensity * m;

	return massDensity * collisionalContributionsToSRDKinematicShearViscosity(
			linearCellSize, meanParticleCountPerCell, srdAngle, timestep);
}

FP AnalyticQuantities::SRDKinematicShearViscosity(
		const FP kT, const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,
		const FP srdAngle, const FP timestep)
{
	const FP nu_kin = kineticContributionsToSRDKinematicShearViscosity(
						kT, m, meanParticleCountPerCell, srdAngle, timestep);
	const FP nu_col = collisionalContributionsToSRDKinematicShearViscosity(
						linearCellSize, meanParticleCountPerCell, srdAngle,	timestep);

	return nu_kin + nu_col;
}

FP AnalyticQuantities::SRDDynamicShearViscosity(
	const FP kT, const FP m, const FP linearCellSize, const FP meanParticleCountPerCell,
	const FP srdAngle, const FP timestep)
{
	const FP mu_kin = kineticContributionsToSRDDynamicShearViscosity(
						kT, m, linearCellSize, meanParticleCountPerCell, srdAngle, timestep);
	const FP mu_col = collisionalContributionsToSRDDynamicShearViscosity(
						m, linearCellSize, meanParticleCountPerCell, srdAngle, timestep);

	return mu_kin + mu_col;
}

FP AnalyticQuantities::approximateSelfDiffusionCoefficient(const unsigned int dimensions, const FP kT, const FP m,
                                                           const FP meanParticleCountPerCell, const FP srdAngle, const FP timestep)
{
	const FP M = meanParticleCountPerCell;

	const FP factor1 = kT * timestep / (2.0 * m);
	const FP term2   = dimensions * M / ( (1 - cos(srdAngle)) * (M - 1 + exp(-M)));

	return factor1 * (term2 - 1);
}

FP AnalyticQuantities::hydrodynamicRadius(const FP kT, const FP dynamicStressViscosity,
                                          const FP selfDiffusionCoefficient)
{
	return kT/(6 * pi * dynamicStressViscosity * selfDiffusionCoefficient);
}
